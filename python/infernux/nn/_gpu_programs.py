"""Built GPU model programs on the common artifact, Kernel and FIFO runtime."""
import hashlib

from infernux import compute
from infernux._compiler.gpu_artifact import _encode_artifact, _read_artifact_payload
from ._programs import ProgramSource, _key


def _forward(program):
    values = [value for value in program._sealed_artifacts.values() if value.autodiff_mode == "none"]
    if len(values) != 1:
        raise ValueError("NN program requires one fixed forward argument contract")
    return values[0]


def _bind_family(program, variants, count):
    whole = _forward(program)
    contract = whole.continuation
    expected = set() if count is None else {"initial", "final"} | ({"resume"} if count > 2 else set())
    if (not contract.eligible or contract.inner_range is not None or not contract.output_parameters
            or contract.output_parameters != contract.fully_written_output_parameters
            or set(variants) != expected):
        raise ValueError("GPU model program has an incomplete continuation family")
    arity = len(whole.argument_layout["parameters"]) - 2
    prefix = [row["kind"] for row in whole.argument_layout["parameters"][:arity]]
    for kind, child in variants.items():
        artifact = _forward(child)
        proof, control = artifact.continuation, artifact.continuation.inner_range
        layouts = artifact.argument_layout["parameters"]
        if (control is None or (control.kind, control.count, control.begin_parameter, control.end_parameter)
                != (kind, count, arity+2, arity+3) or artifact.domain_parameter != whole.domain_parameter
                or len(layouts) != arity+6 or [row["kind"] for row in layouts[:arity]] != prefix
                or [row["kind"] for row in layouts[arity:arity+4]]
                != ["external_buffer", "external_buffer", "int32", "int32"]):
            raise ValueError("GPU model continuation layout differs from its whole program")
        outputs = contract.output_parameters if kind == "final" else (arity+1,)
        if (proof.output_parameters != outputs or proof.fully_written_output_parameters != outputs
                or set(proof.input_parameters) - {arity} - set(contract.input_parameters)):
            raise ValueError("GPU model continuation resource roles differ from its whole program")
    program._inner_variants = variants


def build(programs):
    """Collect actual prepared roles; no source recapture or native submission."""
    from ._primitives import program_family, inner_range

    entries, declarations, files, recipes = {}, {}, {}, {}
    for program in programs:
        for variant in program_family(program):
            roles = {}
            for executable in variant._executables.values():
                artifact = executable.artifact
                payload = _encode_artifact(artifact, executable.specialization)
                identity = hashlib.sha256(payload).hexdigest()
                path = "programs/"+identity+".inxgpu"
                entries[identity] = dict(identity=identity, path=path, bytes=len(payload))
                files[path] = payload
                previous = roles.setdefault(executable.specialization, identity)
                if previous != identity:
                    raise ValueError("GPU program contains ambiguous compiled specializations")
            if not roles or not any(executable.artifact.autodiff_mode == "none" for executable in variant._executables.values()):
                raise RuntimeError("complete GPU cold preparation before building the model")
            declarations[variant] = roles
        control = inner_range(program)
        key = _key(program._nn_recipe)
        row = dict(recipe=program._nn_recipe, roles=declarations[program],
                   variants={kind:declarations[value] for kind,value in program._inner_variants.items()},
                   inner_count=None if control is None else control.count)
        if key not in recipes:
            recipes[key] = row
            continue
        previous = recipes[key]
        if previous['inner_count'] != row['inner_count'] or previous['variants'].keys() != row['variants'].keys():
            raise ValueError("GPU model recipe has conflicting continuation layouts")
        # One occurrence may need local AD while another is stopped. They
        # share the recipe's forward role; collect the actual role closure.
        for old, new in ((previous['roles'], row['roles']),
                         *((previous['variants'][kind], values) for kind, values in row['variants'].items())):
            for specialization, identity in new.items():
                if specialization in old and old[specialization] != identity:
                    raise ValueError("GPU model recipe has ambiguous compiled specializations")
                old[specialization] = identity
    rows = [dict(row, roles=tuple(row['roles'].values()),
                 variants={kind:tuple(roles.values()) for kind,roles in row['variants'].items()})
            for row in recipes.values()]
    return dict(device="gpu", entries=list(entries.values()), recipes=rows), tuple(files.items())


class GpuProgramSource(ProgramSource):
    def __init__(self, record):
        super().__init__(record)
        self.artifacts = {}
        self._bind_index = 0

    def declare_step(self):
        if self._recipe_index == len(self.record["recipes"]):
            return True
        row = self.record["recipes"][self._recipe_index]
        self._check_recipe(row)
        for roles in (row["roles"], *row["variants"].values()):
            if len(set(roles)) != len(roles) or any(i not in self._identities for i in roles):
                raise ValueError("GPU model recipe has duplicate or absent program roles")
        self._recipe_index += 1
        return False

    def decode_entry(self, row, payload):
        """Decode one program from the already checked model container."""
        if len(payload) != row["bytes"]:
            raise ValueError("GPU model program size differs from its catalogue")
        header, artifact = _read_artifact_payload(payload, path=row["path"])
        self.artifacts[row["identity"]] = header["specialization"], artifact

    def _declaration(self, roles, operation):
        values = dict(self.artifacts[identity] for identity in roles)
        if len(values) != len(roles):
            raise ValueError("GPU model has duplicate program specializations")
        result = compute.Kernel._from_artifacts(values, operation)
        artifact = _forward(result)
        result._work_layout = artifact.domain_parameter, artifact.continuation.inner_range
        result._work_access = artifact.continuation.input_parameters, artifact.continuation.output_parameters
        return result

    def bind_step(self):
        if self._bind_index == len(self.record["recipes"]):
            return True
        row = self.record["recipes"][self._bind_index]
        operation = row["recipe"]["operation"]
        program = self._declaration(row["roles"], operation)
        variants = {kind: self._declaration(roles, operation) for kind, roles in row["variants"].items()}
        _bind_family(program, variants, row["inner_count"])
        program._nn_recipe = row["recipe"]
        self.programs[_key(row["recipe"])] = program
        self._bind_index += 1
        return False

    @property
    def native_blocks(self):
        # Retained SPIR-V bytes, including partially decoded input. Kernels
        # borrow these same immutable tasks; they do not create driver pipelines.
        return tuple({("gpu-code", id(task)): len(task)
                      for _, artifact in self.artifacts.values() for task in artifact.spirv_tasks}.items())

    def stage(self):
        files = []
        for row in self.record["entries"]:
            specialization, artifact = self.artifacts[row["identity"]]
            payload = _encode_artifact(artifact, specialization)
            files.append((row["path"], payload))
        return self.record, tuple(files)

    def _discard_unpublished(self):
        for program in self.programs.values():
            for declaration in (program, *program._inner_variants.values()):
                declaration._release_engine_resources()
                declaration._sealed_artifacts.clear()
        self.programs.clear()
        self.artifacts.clear()

    def _publish(self):
        pass  # No OS module or prepared driver pipeline is created by this source.

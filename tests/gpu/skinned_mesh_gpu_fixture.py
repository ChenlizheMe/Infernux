"""One-joint glTF fixture with explicitly authored scale keyframes."""
import base64
import json
from pathlib import Path
import numpy as np

def write_skinned_quad(path, *, envelope=False, translations=None):
    payload = bytearray()
    views, accessors = [], []
    def add(array, kind, component=5126, bounds=False):
        data = np.asarray(array, dtype='<u2' if component == 5123 else '<f4')
        payload.extend(b'\0' * (-len(payload) % 4))
        views.append({'buffer': 0, 'byteOffset': len(payload), 'byteLength': data.nbytes})
        payload.extend(data.tobytes())
        accessor = {'bufferView': len(views)-1, 'componentType': component, 'count': len(data), 'type': kind}
        if bounds:
            accessor.update(min=np.atleast_1d(data.min(axis=0)).tolist(), max=np.atleast_1d(data.max(axis=0)).tolist())
        accessors.append(accessor)
        return len(accessors)-1
    points = [[-.5,-.5,0],[.5,-.5,0],[.5,.5,0],[-.5,.5,0]]
    texture_uvs = [[0,0],[1,0],[1,1],[0,1]]
    triangles = [0,2,1,0,3,2]
    if envelope:
        # Degenerate triangles retain these bounds positions during model
        # import while producing no fragments in any rasterized pass.
        points.extend([[-2,-2,-2],[2,2,2]])
        texture_uvs.extend([[0,0],[0,0]])
        triangles.extend([4,4,4,5,5,5])
    count = len(points)
    position = add(points, 'VEC3', bounds=True)
    normal = add([[0,0,-1]]*count, 'VEC3')
    tangent = add([[1,0,0,1]]*count, 'VEC4')
    uv = add(texture_uvs, 'VEC2')
    joints = add([[0,0,0,0]]*count, 'VEC4', component=5123)
    weights = add([[1,0,0,0]]*count, 'VEC4')
    indices = add(triangles, 'SCALAR', component=5123)
    inverse = add([np.eye(4).flatten().tolist()], 'MAT4')
    times = add([0,1,2,3], 'SCALAR', bounds=True)
    values = translations if translations is not None else [[1,1,1],[2,.5,1],[-2,.5,1],[1,1,1]]
    assert len(values) == 4
    animation_values = add(values, 'VEC3')
    animation_name, animation_path = ('Translate', 'translation') if translations is not None else ('Scale', 'scale')
    document = {'asset': {'version': '2.0'}, 'scene': 0, 'scenes': [{'nodes':[0]}],
        'nodes': [{'name':'Scene','children':[1,2]}, {'name':'Surface','mesh':0,'skin':0}, {'name':'Joint'}],
        'skins': [{'joints':[2], 'inverseBindMatrices':inverse}],
        'meshes':[{'primitives':[{'attributes':{'POSITION':position,'NORMAL':normal,'TANGENT':tangent,
                    'TEXCOORD_0':uv,'JOINTS_0':joints,'WEIGHTS_0':weights}, 'indices':indices}]}],
        'animations':[{'name':animation_name, 'samplers':[{'input':times,'output':animation_values,'interpolation':'LINEAR'}],
            'channels':[{'sampler':0,'target':{'node':2,'path':animation_path}}]}],
        'buffers':[{'byteLength':len(payload),'uri':'data:application/octet-stream;base64,'+base64.b64encode(payload).decode('ascii')}],
        'bufferViews': views, 'accessors':accessors}
    Path(path).write_text(json.dumps(document), encoding='utf-8')

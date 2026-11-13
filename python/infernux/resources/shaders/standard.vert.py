import infernux.shader as shader


class Standard(shader.vert):
    shader_id = "infernux/standard"

    def deform(self, vertex: shader.VertexData, ctx: shader.VertexContext) -> shader.VertexData:
        return vertex

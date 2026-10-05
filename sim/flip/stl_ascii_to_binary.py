"""Convert an ASCII STL to binary STL, the only kind MuJoCo can load.

    python -m sim.flip.stl_ascii_to_binary IN.stl OUT.stl
"""
import argparse
import re
import struct

FACET = re.compile(r"facet normal\s+(\S+)\s+(\S+)\s+(\S+)\s+outer loop(.*?)endloop", re.S)
VERTEX = re.compile(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)")


def convert(src: str, dst: str) -> int:
    facets = []
    for *normal, body in FACET.findall(open(src).read()):
        vertices = VERTEX.findall(body)
        if len(vertices) != 3:
            raise ValueError(f"{src}: facet with {len(vertices)} vertices")
        facets.append((normal, vertices))
    with open(dst, "wb") as f:
        f.write(b"\0" * 80 + struct.pack("<I", len(facets)))
        for normal, vertices in facets:
            f.write(struct.pack("<3f", *map(float, normal)))
            for v in vertices:
                f.write(struct.pack("<3f", *map(float, v)))
            f.write(b"\0\0")
    return len(facets)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    args = p.parse_args()
    print(f"{convert(args.src, args.dst)} triangles -> {args.dst}")

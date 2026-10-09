"""Small synthetic VRC2 archive test without dataset or PyTorch downloads."""
import json
import struct

import numpy as np
import zstandard as zstd


weights = np.linspace(-0.75, 0.75, 64, dtype=np.float32).reshape(2, 32)
scales = np.maximum(np.max(np.abs(weights), axis=1) / 127, np.finfo(np.float32).tiny).astype('<f4')
codes = np.clip(np.rint(weights / scales[:, None]), -127, 127).astype(np.int16)
parts = [scales.tobytes(), (codes + 128).astype(np.uint8).tobytes(), np.array([0.25, -0.25], dtype=np.float32).tobytes()]
specs = [dict(name='linear.weight', shape=[2, 32], mode='int8', parts=2),
         dict(name='linear.bias', shape=[2], mode='raw', dtype='float32', parts=1)]
header = json.dumps(dict(tensors=specs, group=32, lengths=[len(part) for part in parts]), separators=(',', ':')).encode()
archive = zstd.ZstdCompressor(level=3).compress(b'VRC2' + struct.pack('<I', len(header)) + header + b''.join(parts))
blob = zstd.ZstdDecompressor().decompress(archive)
assert blob[:4] == b'VRC2'
header_size = struct.unpack_from('<I', blob, 4)[0]
metadata = json.loads(blob[8:8 + header_size])
assert metadata['group'] == 32 and [item['name'] for item in metadata['tensors']] == ['linear.weight', 'linear.bias']
offset = 8 + header_size
lengths = metadata['lengths']
read_scales = np.frombuffer(blob, dtype='<f4', count=lengths[0] // 4, offset=offset)
offset += lengths[0]
read_codes = np.frombuffer(blob, dtype=np.uint8, count=lengths[1], offset=offset)
offset += lengths[1]
read_bias = np.frombuffer(blob, dtype=np.float32, count=2, offset=offset)
decoded = {'linear.weight': ((read_codes.astype(np.int16) - 128).astype(np.float32).reshape(2, 32) * read_scales[:, None]),
           'linear.bias': read_bias}
expected = (codes.astype(np.float32) * scales[:, None]).reshape(2, 32)
assert np.array_equal(decoded['linear.weight'], expected)
assert np.array_equal(decoded['linear.bias'], np.array([0.25, -0.25], dtype=np.float32))
print(f'PASS: {len(decoded)} tensors, {len(archive)} archive bytes, exact decoded values')

# NAFNet Deblur ONNX Artifacts

`nafnet_deblur.onnx` uses ONNX external data. Before loading the model, place
the weights file at `nafnet_deblur-onnx-w8a16/nafnet_deblur.data`.

- Size: `272800256` bytes
- SHA-256: `742aaccef608e9f4380dbb940994498dc0c50ca4bb9938b89e0109513321d004`

Verify the local file with:

```bash
sha256sum nafnet_deblur-onnx-w8a16/nafnet_deblur.data
```

The weights file is intentionally excluded from Git because it exceeds the
hosting service's per-file limit.

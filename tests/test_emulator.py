from VeryAccurateEmulator import emulator

def test_gen_model():
    in_dim = 7
    hidden_dims = [32, 64, 256]
    out_dim = 451
    model = emulator._gen_model(in_dim, hidden_dims, out_dim, "relu")
    all_dims = [in_dim] + hidden_dims + [out_dim]
    assert len(model.layers) == len(all_dims)
    for i, layer in enumerate(model.layers):
        if i == 0:
            shape = layer.output_shape[0][-1]
        else:
            shape = layer.output_shape[-1]
        assert shape == all_dims[i]

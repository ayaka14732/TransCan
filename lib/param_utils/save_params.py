from flax.serialization import msgpack_serialize, msgpack_restore

from .assert_tree_equal import assert_tree_equal

def save_params(params, filename, roundtrip_check=True):
    serialized_params = msgpack_serialize(params)

    if roundtrip_check:
        recovered_params = msgpack_restore(serialized_params)
        assert_tree_equal(params, recovered_params)

    with open(filename, 'wb') as f:
        f.write(serialized_params)

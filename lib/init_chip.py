import os

def find_free_port() -> int:
    import socket
    from contextlib import closing
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def init_one_chip(process_num) -> None:
    port = find_free_port()
    os.environ['TPU_CHIPS_PER_HOST_BOUNDS'] = '1,1,1'
    os.environ['TPU_HOST_BOUNDS'] = '1,1,1'
    # Different per process:
    os.environ['TPU_VISIBLE_DEVICES'] = str(process_num)  # '0', '1', '2', '3'
    # Pick a unique port per process
    os.environ['TPU_MESH_CONTROLLER_ADDRESS'] = f'localhost:{port}'
    os.environ['TPU_MESH_CONTROLLER_PORT'] = str(port)

def init_two_chip() -> None:
    port = find_free_port()
    os.environ['TPU_CHIPS_PER_HOST_BOUNDS'] = '1,2,1'
    os.environ['TPU_HOST_BOUNDS'] = '1,1,1'
    # Different per process:
    os.environ['TPU_VISIBLE_DEVICES'] = '0,1'  # '2,3'
    # Pick a unique port per process
    os.environ['TPU_MESH_CONTROLLER_ADDRESS'] = f'localhost:{port}'
    os.environ['TPU_MESH_CONTROLLER_PORT'] = str(port)

def init_four_chip() -> None:
    os.environ['TPU_CHIPS_PER_HOST_BOUNDS'] = '2,2,1'
    os.environ['TPU_HOST_BOUNDS'] = '1,1,1'
    os.environ['TPU_VISIBLE_DEVICES'] = '0,1,2,3'

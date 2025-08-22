import multiprocessing as mp, torch, cuda_ipc_kernel_ext as ipc

N = 4
def worker(rank, conn):
    dev = rank
    torch.cuda.set_device(dev)
    src: torch.Tensor = (rank * N + torch.arange(N, dtype=torch.float32, device=f"cuda:{dev}")).contiguous()
    
    print(f"rank{rank}: src={src.tolist()} offset={src.storage_offset()} elems {src.storage_offset() * 4} bytes")

    handle, off = ipc.export_ipc_handle_and_offset(src)
    print(f"rank{rank}: offset={off}")

    conn.send((handle, off, N))
    peer_handle, peer_off, peer_n = conn.recv()

    other = 1 - rank
    ipc.enable_peer_access(dev, other)

    base = ipc.open_remote_base(peer_handle)
    peer_ptr = ipc.add_offset(base, peer_off)

    dst = torch.empty_like(src)
    ipc.copy_from_remote(dst, peer_ptr, peer_n)
    print(f"rank{rank}: received {dst.tolist()}")

    ipc.add_inplace_remote(peer_ptr, 10.0 + rank, peer_n, dev)
    torch.cuda.synchronize()

    print(f"rank{rank}: after adding {10.0 + rank} = {dst.tolist()}")

    ipc.copy_from_remote(dst, peer_ptr, peer_n)  # re-read to show the change

    conn.send("done"); _ = conn.recv()
    print(f"[rank{rank}] src after peer write: {src.tolist()}")

    ipc.close_remote_base(base)
    conn.close()

if __name__ == "__main__":
    assert torch.cuda.device_count() >= 2
    mp.set_start_method("spawn", force=True)
    a, b = mp.Pipe(duplex=True)
    mp.Process(target=worker, args=(0, a)).start()
    mp.Process(target=worker, args=(1, b)).start()

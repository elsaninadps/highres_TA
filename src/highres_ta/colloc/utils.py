def extract_tar(fname_tar, dest_dir):
    import pathlib
    import tarfile

    dest_dir = pathlib.Path(dest_dir)

    tar_obj = tarfile.open(fname_tar)
    tar_obj.extractall(path=dest_dir)

    member0 = tar_obj.getnames()[0]
    out_paths = [str(p) for p in (dest_dir / member0).glob("*")]

    return out_paths


def compress_tar(dir_name):
    import pathlib
    import tarfile

    dir_name = pathlib.Path(dir_name)
    tar_name = dir_name.with_suffix(".tar")

    with tarfile.open(tar_name, mode="w:") as tar_obj:
        tar_obj.add(dir_name)

    return tar_name

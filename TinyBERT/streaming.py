import os
import logging
import fnmatch
import random
import numpy as np
import tensorflow as tf
import subprocess


def get_stat(dirname, filename_pat="*"):
    if not tf.io.gfile.exists(dirname):
        logging.warning(f"{dirname} does not exist!")
        return None

    stat = {}
    for x in tf.io.gfile.listdir(dirname):
        if fnmatch.fnmatch(x, filename_pat):
            file = os.path.join(dirname, x)
            result = subprocess.getoutput(f'wc -l {file}')
            size = int(result.split(' ')[0])
            stat[file] = size
    return stat


def get_files(dirname, filename_pat="*", recursive=False):
    if not tf.io.gfile.exists(dirname):
        logging.warning(f"{dirname} does not exist!")
        return None
    files = []
    for x in tf.io.gfile.listdir(dirname):
        path = os.path.join(dirname, x)
        if tf.io.gfile.isdir(path):
            if recursive:
                files.extend(get_files(path, filename_pat))
        elif fnmatch.fnmatch(x, filename_pat):
            files.append(path)
    return files


def get_worker_files(dirname,
                     worker_rank,
                     world_size,
                     filename_pat="*",
                     shuffle=False,
                     seed=0):
    """Get file paths belong to one worker."""
    all_files = get_files(dirname, filename_pat)
    all_files.sort()
    if shuffle:
        random.seed(seed)
        random.shuffle(all_files)
    files = []
    for i in range(worker_rank, len(all_files), world_size):
        files.append(all_files[i])
    logging.info(
        f"worker_rank:{worker_rank}, world_size:{world_size}, shuffle:{shuffle}, seed:{seed}, directory:{dirname}, files:{files}"
    )
    return files


class StreamReader:
    def __init__(self, data_paths, batch_size, shuffle=False, shuffle_buffer_size=1000):
        tf.config.experimental.set_visible_devices([], device_type="GPU")
        path_len = len(data_paths)
        dataset = tf.data.Dataset.list_files(data_paths).interleave(
            lambda x: tf.data.TextLineDataset(x),
            cycle_length=path_len,
            block_length=128,
            num_parallel_calls=min(path_len, 64),
        )

        if shuffle:
            dataset = dataset.shuffle(
                shuffle_buffer_size, reshuffle_each_iteration=True)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)
        self.next_batch = dataset.make_one_shot_iterator().get_next()
        self.session = None

    def reset(self):
        if self.session:
            self.session.close()
        self.session = tf.Session()
        self.endofstream = False

    def get_next(self):
        try:
            ret = self.session.run(self.next_batch)
        except tf.errors.OutOfRangeError:
            self.endofstream = True
            return None
        return ret

    def reach_end(self):
        return self.endofstream


class StreamSampler:
    def __init__(
        self,
        data_dir,
        filename_pat,
        batch_size,
        worker_rank,
        world_size,
        enable_shuffle=False,
        shuffle_buffer_size=1000,
        shuffle_seed=0,
    ):
        data_paths = get_worker_files(
            data_dir,
            worker_rank,
            world_size,
            filename_pat,
            shuffle=enable_shuffle,
            seed=shuffle_seed,
        )
        self.stream_reader = StreamReader(
            data_paths,
            batch_size,
            enable_shuffle,
            shuffle_buffer_size
        )

    def __iter__(self):
        self.stream_reader.reset()
        return self

    def __next__(self):
        """Implement iterator interface."""
        next_batch = self.stream_reader.get_next()
        if not isinstance(next_batch, np.ndarray) and not isinstance(
                next_batch, tuple):
            raise StopIteration
        return next_batch

    def reach_end(self):
        return self.stream_reader.reach_end()

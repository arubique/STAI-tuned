import os
from torch.utils.data import DataLoader, SubsetRandomSampler
import shutil
import gdown
import requests
import random
import pickle
import traceback
import sys
import os
import torch
from typing import Union, Dict, List, Callable
from collections import UserDict
import warnings
import gc
import numpy as np
from stnd.utility.data_utils import make_or_load_from_cache


# local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utility.logger import make_logger
from utility.utils import (
    remove_filename_extension,
    remove_file_or_folder,
    remove_all_but_subdirs,
    move_folder_contents,
    compute_file_hash,
    get_hash,
    log_or_print,
    error_or_print,
    get_project_root_path,
    prepare_for_pickling,
    randomly_subsample_indices_uniformly,
    deterministically_subsample_indices_uniformly,
    show_images,
    append_dict,
    compute_proportion,
    add_custom_properties,
    raise_unknown,
    get_with_assert,
    load_from_pickle,
)
from utility.imports import FROM_CLASS_KEY, make_from_class_ctor

sys.path.pop(0)


CHECK_FILE_HASH = False


EMPTY_URL = "¯\_(ツ)_/¯"
YANDEX_API_ENDPOINT = (
    "https://cloud-api.yandex.net/v1/disk"
    + "/public/resources/download?public_key={}"
)


H5_EXTENSION = ".h5"


TRAIN_KEY = "train"
VAL_KEY = "val"


SCALING_FACTOR = 255


class DatasetWrapperWithTransforms(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        assert isinstance(transform, Callable)
        self.inner_dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.inner_dataset[index]
        x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.inner_dataset)


def make_dataset_wrapper_with_transforms(dataset, transform):
    return DatasetWrapperWithTransforms(dataset, transform)


def make_default_data_path():
    return os.path.join(get_project_root_path(), "data")


def fetch_data(
    dataset_path,
    assert_folder_func,
    do_fetch_data_func,
    logger=None,
    dataset_path_is_folder=True,
):
    """
    Makes sure that dataset is under the <dataset_path> path
    """
    dataset_name = remove_filename_extension(
        os.path.basename(dataset_path), must_have_extension=False
    )
    log_or_print(
        'Verifying "{}" dataset..'.format(dataset_name),
        logger=logger,
        auto_newline=True,
    )
    if not assert_folder_func(dataset_path):
        log_or_print(
            'Could not verify "{}" dataset in {}'.format(
                dataset_name, dataset_path
            ),
            logger=logger,
            auto_newline=True,
        )
        if os.path.exists(dataset_path):
            object_type = "folder" if dataset_path_is_folder else "file"
            log_or_print(
                "Removing existing {} {}..".format(object_type, dataset_path),
                logger=logger,
                auto_newline=True,
            )
            remove_file_or_folder(dataset_path)
        os.makedirs(
            dataset_path
            if dataset_path_is_folder
            else os.path.dirname(dataset_path),
            exist_ok=True,
        )
        do_fetch_data_func(dataset_name, dataset_path, logger=logger)
        log_or_print(
            'Verifying "{}" dataset '
            "downloaded into {}..".format(dataset_name, dataset_path),
            logger=logger,
            auto_newline=True,
        )
        if not assert_folder_func(dataset_path):
            raise Exception("Downloaded data is not what was expected.")
    else:
        log_or_print(
            '"{}" is already in {}!'.format(dataset_name, dataset_path),
            logger=logger,
            auto_newline=True,
        )


def convert_dataset_to_tensor(dataset):
    data_loader = DataLoader(dataset, batch_size=len(dataset))
    return next(iter(data_loader))


def convert_dataset_to_tensor_generator(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size)


def make_contents_as_in_subfolder(src_folder, subdir_name):
    remove_all_but_subdirs(src_folder, [subdir_name])
    folder_to_keep = os.path.join(src_folder, subdir_name)
    move_folder_contents(folder_to_keep, src_folder)
    shutil.rmtree(folder_to_keep)


def do_fetch_dataset_as_file(url):
    def do_fetch_file_by_url(dataset_name, dataset_path, logger=make_logger()):
        if url == EMPTY_URL:
            NotImplementedError()
        log_or_print(
            'Downloading "{}" into {}..'.format(dataset_name, dataset_path),
            logger=logger,
            auto_newline=True,
        )
        download_file_by_link(url, dataset_path)

    return do_fetch_file_by_url


def download_file_by_link(url, path):
    if "google" in url:
        gdown.download(url, path, quiet=True)
    elif "yandex" in url:
        download_file_by_yadisk_link(url, path)
    else:
        raise Exception(
            "Can not download by url which is not from Yandex "
            "or Google drive"
        )


def download_file_by_yadisk_link(sharing_link, filename):
    def _get_real_direct_link(sharing_link):
        pk_request = requests.get(YANDEX_API_ENDPOINT.format(sharing_link))
        return pk_request.json().get("href")

    direct_link = _get_real_direct_link(sharing_link)
    if direct_link:
        download = requests.get(direct_link)
        with open(filename, "wb") as out_file:
            out_file.write(download.content)
    else:
        raise Exception("Can not download file from {}".format(sharing_link))


def assert_dataset_as_file(file_hash, check_file_hash=CHECK_FILE_HASH):
    def assert_file_by_hash(dataset_path):
        path_exists = os.path.exists(dataset_path)

        file_hash_is_correct = True
        if path_exists and check_file_hash:
            file_hash_is_correct = compute_file_hash(dataset_path) == file_hash

        return path_exists and file_hash_is_correct

    return assert_file_by_hash


def randomly_subsampled_dataloader(dataloader, fraction, batch_size=None):
    if isinstance(dataloader, ManyDataloadersWrapper):
        new_dataloaders_list = []
        for sub_dataloader in dataloader.dataloaders_list:
            new_dataloaders_list.append(
                randomly_subsampled_dataloader(
                    sub_dataloader, fraction, batch_size=batch_size
                )
            )
        new_dataloader = wrap_dataloader(
            new_dataloaders_list, dataloader.wrapper
        )
    else:
        new_dataloader = subsample_dataloader_randomly(
            dataloader, fraction, batch_size=batch_size
        )

    return new_dataloader


def subsample_dataloader_randomly(dataloader, fraction, batch_size=None):
    num_samples = compute_proportion(fraction, len(dataloader.dataset))
    dataloader_is_wrapper = False
    wrapper = None
    # TODO(Alex | 12.06.2023) make valid for DspritesDataloaderWrapper as well
    if isinstance(dataloader, SingleDataloaderWrapper):
        dataloader_is_wrapper = True
        wrapper = dataloader.wrapper
        dataloader = dataloader.dataloader
    elif isinstance(dataloader, ManyDataloadersWrapper):
        raise TypeError(
            f"Can't subsample dataloader of type: {type(dataloader)}"
        )
    else:
        assert isinstance(dataloader, DataLoader)

    subsampled_indices = random.sample(
        list(range(len(dataloader.dataset))), num_samples
    )

    dataloader_init_args = get_dataloader_init_args_from_existing_dataloader(
        dataloader
    )

    dataloader_init_args["sampler"] = SubsetRandomSampler(subsampled_indices)
    if batch_size:
        dataloader_init_args["batch_size"] = batch_size

    new_dataloader = DataLoader(**dataloader_init_args)

    add_custom_properties(dataloader, new_dataloader)

    if dataloader_is_wrapper:
        if wrapper is not None:
            new_dataloader = wrap_dataloader(new_dataloader, wrapper=wrapper)
        else:
            raise Exception(
                "Most likely DspritesDataloaderWrapper is used, "
                "but it is not supported."
            )
    return new_dataloader


def get_dataloader_init_args_from_existing_dataloader(
    dataloader, copy_batch_sampler=False
):
    dataloader_init_args = {
        "dataset": dataloader.dataset,
        "num_workers": dataloader.num_workers,
        "collate_fn": dataloader.collate_fn,
        "pin_memory": dataloader.pin_memory,
        "timeout": dataloader.timeout,
        "worker_init_fn": dataloader.worker_init_fn,
        "prefetch_factor": dataloader.prefetch_factor,
        "persistent_workers": dataloader.persistent_workers,
    }
    if copy_batch_sampler:
        dataloader_init_args["batch_sampler"] = dataloader.batch_sampler
    else:
        dataloader_init_args["sampler"] = dataloader.sampler
        dataloader_init_args["batch_size"] = dataloader.batch_size
        dataloader_init_args["drop_last"] = dataloader.drop_last
    return dataloader_init_args


class SingleDataloaderWrapper:
    def __init__(self, dataloader, wrapper, **kwargs):
        self.dataloader = dataloader
        self.kwargs = kwargs
        self.dataset = self.dataloader.dataset
        self.wrapper = wrapper
        self.batch_size = self.dataloader.batch_size

    def __iter__(self):
        return self.wrapper(iter(self.dataloader), **self.kwargs)

    def __len__(self):
        return len(self.dataloader)


class ManyDataloadersWrapper:
    def __init__(self, dataloaders_list, wrapper, **kwargs):
        self.dataloaders_list = dataloaders_list
        self.batch_size = None
        self.length = 0
        self.wrapper = wrapper
        self.kwargs = kwargs
        for dataloader in dataloaders_list:
            if self.batch_size is None:
                self.batch_size = dataloader.batch_size
            else:
                assert (
                    self.batch_size == dataloader.batch_size
                ), "All chaining dataloaders should have the same batchsize"
            self.length += len(dataloader)

    def __iter__(self):
        return self.wrapper(
            [iter(dataloader) for dataloader in self.dataloaders_list],
            **self.kwargs,
        )

    def __len__(self):
        return self.length


def wrap_dataloader(wrappable, wrapper, **kwargs):
    if isinstance(wrappable, list):
        return ManyDataloadersWrapper(wrappable, wrapper, **kwargs)
    else:
        wrapped_dataloader = SingleDataloaderWrapper(
            wrappable, wrapper, **kwargs
        )
        add_custom_properties(wrappable, wrapped_dataloader)
        return wrapped_dataloader


def get_generic_train_eval_dataloaders(
    train_datasets_dict,
    eval_datasets_dict,
    train_batch_size,
    eval_batch_size,
    shuffle_train=True,
    shuffle_eval=False,
    **dataloader_kwargs,
):
    def add_dataloaders(
        datasets_dict, batch_size, shuffle, **dataloader_kwargs
    ):
        dataloaders_dict = {}
        for dataset_name, dataset in datasets_dict.items():
            dataloaders_dict[dataset_name] = DataLoader(
                dataset,
                shuffle=shuffle,
                batch_size=batch_size,
                **dataloader_kwargs,
            )
        return dataloaders_dict

    train_dataloaders = None
    eval_dataloaders = None

    if train_batch_size > 0:
        train_dataloaders = add_dataloaders(
            train_datasets_dict,
            train_batch_size,
            shuffle_train,
            **dataloader_kwargs,
        )

    if eval_batch_size > 0:
        eval_dataloaders = add_dataloaders(
            eval_datasets_dict,
            eval_batch_size,
            shuffle_eval,
            **dataloader_kwargs,
        )

    return train_dataloaders, eval_dataloaders


def uniformly_subsample_dataset(dataset, num_samples, deterministic):
    if num_samples > 0:
        if deterministic:
            indices = deterministically_subsample_indices_uniformly(
                len(dataset), num_samples
            )
        else:
            indices = randomly_subsample_indices_uniformly(
                len(dataset), num_samples
            )
        dataset = torch.utils.data.Subset(dataset, indices)
    return dataset


def show_dataloader_first_batch(
    dataloader: torch.utils.data.DataLoader, label_names: List[str]
):
    """
    Plots first batch of a dataloader
    using local function "show_images_batch".
    Args:
        dataloader (torch.utils.data.DataLoader): a dataloader
            which first batch is shown.
        label_names (List[str]): list of names for each label.
            For example, if a dataloader generates
            tuple (input, List[label_0, ... label_k]),
            label_names will be List[label_0_name, ..., label_k_name].
    """

    iterator_element = next(iter(dataloader))
    assert len(iterator_element) > 1
    if len(iterator_element) == 2:
        images_batch, labels_batch = iterator_element
    else:
        images_batch = iterator_element[0]
        labels_batch = list(iterator_element[1:])

    images_batch = images_batch.cpu()

    if isinstance(labels_batch, list):
        for i in range(len(labels_batch)):
            labels_batch[i] = labels_batch[i].cpu()
    else:
        labels_batch = labels_batch.cpu()

    labels_batch = make_named_labels_batch(label_names, labels_batch)

    show_images_batch(images_batch, labels_batch)


def show_images_batch(
    images_batch: torch.tensor,
    label_batches: Union[torch.tensor, Dict[str, torch.tensor]] = None,
):
    """
    Shows a batch of images as a square image grid.
    If <label_batches> is provided, each image title consists of label names
    and their corresponding values.
    Args:
        images_batch (torch.tensor): of images batch.
        label_batches (Union[torch.tensor, Dict[str, torch.tensor]], optional):
            information about labels that is either a label batch
            or a dict that maps label name to the corresponding label batch.
    """

    images_list = []

    images_batch = images_batch.cpu()

    label_lists = None if label_batches is None else {}

    n_images = images_batch.shape[0]

    if label_batches is not None:
        if not isinstance(label_batches, dict):
            label_batches = {"label": label_batches}

        for label_batch in label_batches.values():
            assert label_batch.shape[0] == n_images

    for i in range(n_images):
        images_list.append(images_batch[i])
        if label_lists is not None:
            for label_name, label_batch in label_batches.items():
                append_dict(
                    label_lists,
                    {label_name: label_batch[i].item()},
                    allow_new_keys=True,
                )

    show_images(images_list, label_lists)


def make_named_labels_batch(label_names, labels_batch):
    if isinstance(labels_batch, list):
        assert len(labels_batch) == len(label_names)
        labels_batch = {
            label_name: label_batch
            for label_name, label_batch in zip(label_names, labels_batch)
        }
    else:
        assert len(label_names) == 1
        labels_batch = {label_names[0]: labels_batch}

    return labels_batch


class ChainingIteratorsWrapper:
    def __init__(self, iterators_list, random_order=False):
        self.iterators = iterators_list
        self.not_exhausted_ids = list(range(len(self.iterators)))
        self.random_order = random_order
        self._next_id()

    def __iter__(self):
        return self

    def _next_id(self):
        if self.random_order:
            self.current_iterator_id = random.choice(self.not_exhausted_ids)
        else:
            self.current_iterator_id = self.not_exhausted_ids[0]

    def __next__(self):
        self._next_id()
        current_iterator = self.iterators[self.current_iterator_id]
        try:
            iteration_element = next(current_iterator)
        except StopIteration:
            if len(self.not_exhausted_ids) == 1:
                raise
            else:
                index_to_pop = self.not_exhausted_ids.index(
                    self.current_iterator_id
                )
                self.not_exhausted_ids.pop(index_to_pop)
                iteration_element = self.__next__()
        return iteration_element


def chain_dataloaders(dataloaders_list, random_order=False):
    assert len(dataloaders_list) > 1, "Need at least 2 dataloaders to chain"
    return wrap_dataloader(
        dataloaders_list, ChainingIteratorsWrapper, random_order=random_order
    )


class DropLastIteratorWrapper:
    def __init__(self, iterator):
        self.iterator = iterator

    def __iter__(self):
        return self

    def __next__(self):
        next_item = next(self.iterator)
        return next_item[:-1]


def make_sampler(data_source, sampler_config):
    if sampler_config is None:
        return None

    sampler_type = get_with_assert(sampler_config, "type")
    specific_sampler_config = get_with_assert(sampler_config, sampler_type)
    if sampler_type.startswith(FROM_CLASS_KEY):
        return make_from_class_ctor(specific_sampler_config, [data_source])
    else:
        raise_unknown("sampler type", sampler_type, "sampler config")


def get_base_dataset(
    base_data_config,
    dataset_filename,
    get_dataset_hash,
    get_dataset_url,
    read_dataset,
    check_file_hash=CHECK_FILE_HASH,
    logger=None,
):
    assert "." in dataset_filename
    dataset_name, _ = dataset_filename.split(".")

    log_or_print(f"Making base_data for {dataset_name}..", logger=logger)
    dataset_path = os.path.join(
        get_with_assert(base_data_config, "path"), dataset_filename
    )
    fetch_data(
        dataset_path=dataset_path,
        assert_folder_func=assert_dataset_as_file(
            get_dataset_hash(dataset_name), check_file_hash=check_file_hash
        ),
        do_fetch_data_func=do_fetch_dataset_as_file(
            get_dataset_url(dataset_name)
        ),
        logger=logger,
        dataset_path_is_folder=False,
    )
    dataset = read_dataset(filename=dataset_path)

    return dataset


class CachingDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, unique_hash, cache_path):
        self.dataset = dataset
        self.unique_hash = unique_hash
        self.cache_path = cache_path

    def __getitem__(self, index):
        def make_item(item_config, logger):
            return {"item": self.dataset[index]}

        item = make_or_load_from_cache(
            "dataset_item_{}".format(index),
            object_config={},
            unique_hash=self.unique_hash,
            make_func=make_item,
            cache_path=self.cache_path,
            verbose=False,
        )

        return item["item"]

    def __len__(self):
        return len(self.dataset)


def make_caching_dataset(dataset, unique_hash, cache_path):
    return CachingDatasetWrapper(dataset, unique_hash, cache_path)


class DatasetWrapperWithIndex(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return *self.dataset[index], index

    def __len__(self):
        return len(self.dataset)


def make_dataset_wrapper_with_index(dataset):
    return DatasetWrapperWithIndex(dataset)


# function is taken from: https://github.com/bethgelab/model-vs-human/blob/master/modelvshuman/models/pytorch/simclr/utils/modules.py#L21
def unnormalize(tensor, mean=[0], std=[1], inplace=False):
    """Unnormalize a tensor image by first multiplying by std (channel-wise) and then adding the mean (channel-wise)

    Args:
        tensor (Tensor): Tensor image of size (N, C, H, W) to be de-standarized.
        mean (sequence): Sequence of original means for each channel.
        std (sequence): Sequence of original standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Unnormalized Tensor image.

    """

    if not torch.is_tensor(tensor):
        raise TypeError(
            "tensor should be a torch tensor. Got {}.".format(type(tensor))
        )

    if tensor.ndimension() != 4:
        raise ValueError(
            "Expected tensor to be a tensor image of size (N, C, H, W). Got tensor.size() = "
            "{}.".format(tensor.size())
        )
    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)

    if (std == 0).any():
        raise ValueError(
            "std evaluated to zero after conversion to {}, leading to division by zero.".format(
                dtype
            )
        )

    if mean.ndim == 1:
        mean = mean[None, :, None, None]
    if std.ndim == 1:
        std = std[None, :, None, None]

    tensor.mul_(std).add_(mean)
    return tensor


def tensor_for_matplotlib(tensor):
    if torch.is_tensor(tensor):
        tensor = tensor.cpu().numpy()

    if len(tensor.shape) == 4:
        assert tensor.shape[0] == 1
        tensor = tensor.squeeze(0)

    assert len(tensor.shape) == 3

    if tensor.shape[0] in (1, 3):
        tensor = np.transpose(tensor, (1, 2, 0))

    return tensor

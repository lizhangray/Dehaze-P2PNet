# build dataset according to given 'dataset_file'
def build_dataset(args):
    if args.dataset_file == 'Hazy_SHTA':
        from crowd_datasets.Hazy_ShanghaiTech.loading_data_A import loading_data
        return loading_data
    if args.dataset_file == 'Hazy_SHTB':
        from crowd_datasets.Hazy_ShanghaiTech.loading_data_B import loading_data
        return loading_data
    if args.dataset_file == 'Hazy_SHARGBD':
        from crowd_datasets.Hazy_ShanghaiTechRGBD.loading_data import loading_data
        return loading_data
    if args.dataset_file == 'Rainy_SHARGBD':
        from crowd_datasets.Rainy_ShanghaiTechRGBD.loading_data import loading_data
        return loading_data
    if args.dataset_file == 'Hazy_JHU':
        from crowd_datasets.Hazy_JHU.loading_data import loading_data
        return loading_data
    return None

from apply_artifacts import ArtifactBatchProcessor

# src_folder = '~/CrowdsourcingDataset-Amgadetal2019/images'
# overlay_folders = {
#     'bubble': ('~/unet_v1.0/artifacts/imgs/bubble', 1),
#     'cell': ('~/unet_v1.0/artifacts/imgs/cell', 1),
#     'dark_spots': ('~/unet_v1.0/artifacts/imgs/dark_spots', 1),
#     'fat': ('~/unet_v1.0/artifacts/imgs/fat', 1),
#     'group': ('~/unet_v1.0/artifacts/imgs/group', 1),
#     'squamous': ('~/unet_v1.0/artifacts/imgs/squamous', 1),
#     'threads': ('~/unet_v1.0/artifacts/imgs/threads', 0)
#     ### CAUTION: threads could throw RuntimeError: The size of tensor a (516) must match the size of tensor b (0) at non-singleton dimension 1
# }
# output_folder = '~/unet_v1.0/artifacts/output'
# # output_folder = '~/unet_v1.0/artifacts/output_mini'

# processor = ArtifactBatchProcessor(src_folder, overlay_folders, output_folder)
# processor.process_all_images()






src_folder = '~/unet_v1.0/data_train_2/images'
custom_overlay_folders = {
    'bubble': 0, 
    'cell': 1,
    'dark_spots': 1,
    'group': 1,
    'fat': 1,
    'squamous': 1,
    'threads': 0
}
output_folder = '~/unet_v1.0/data_train_2_augmented/images'

# Using custom overlay folders with default paths but custom counts
processor = ArtifactBatchProcessor(src_folder, overlay_folders=custom_overlay_folders, output_folder=output_folder)
processor.process_all_images()

import torch
import numpy as np
import cv2 as cv
from artifacts import ArtifactAugmentation
import imageio.v2 as imageio  # use imageio.v2 to avoid DeprecationWarning
import os
import random



# ################## arbitrary number of overlays from different categories ##################
# class ArtifactBatchProcessor:
#     def __init__(self, src_folder, overlay_folders, output_folder):
#         self.src_folder = os.path.expanduser(src_folder)
#         # expanduser in case os doesn't recognize relative paths and raise FileNotFoundError
#         self.overlay_folders = {k: (os.path.expanduser(v[0]), v[1]) for k, v in overlay_folders.items()}
#         self.output_folder = os.path.expanduser(output_folder)
#         self.augmenter = ArtifactAugmentation()
        
#         self._check_directories()

#     def _check_directories(self):
#         if not os.path.exists(self.src_folder) or not os.path.isdir(self.src_folder):
#             raise FileNotFoundError(f"The source folder '{self.src_folder}' does not exist.")
#         for folder, _ in self.overlay_folders.values():
#             if not os.path.exists(folder) or not os.path.isdir(folder):
#                 raise FileNotFoundError(f"The overlay folder '{folder}' does not exist.")
#         if not os.path.exists(self.output_folder):
#             os.makedirs(self.output_folder)

#     def _process_image(self, src_image_path, overlay_image_paths):
#         src = imageio.imread(src_image_path)
#         mask = np.ones((src.shape[0], src.shape[1]), dtype=np.float32)

#         src_tensor = torch.from_numpy(src).permute(2, 0, 1).float() / 255.0  # (H, W, C) -> (C, H, W)
#         mask_tensor = torch.from_numpy(mask).float()

#         for overlay_image_path in overlay_image_paths:
#             overlay = imageio.imread(overlay_image_path)
#             overlay_mask = cv.cvtColor(overlay[:, :, :3], cv.COLOR_RGB2GRAY)
#             overlay_mask = cv.resize(overlay_mask, (src.shape[1], src.shape[0]))

#             augmented_image, updated_mask = self.augmenter.transparentOverlay(
#                 src=src_tensor,
#                 mask=mask_tensor,
#                 scale=2,
#                 mask_threshold=0.3,
#                 overlay_path=overlay_image_path,
#             )

#             src_tensor = augmented_image  # Update the src_tensor with the newly augmented image
#             mask_tensor = updated_mask  # Update the mask_tensor with the updated mask

#         augmented_image = augmented_image.permute(1, 2, 0).numpy() * 255.0
#         updated_mask = updated_mask.numpy() * 255.0

#         base_name = os.path.basename(src_image_path)
#         augmented_image_path = os.path.join(self.output_folder, f"augmented_{base_name}")
#         updated_mask_path = os.path.join(self.output_folder, f"mask_{base_name}")

#         imageio.imwrite(augmented_image_path, augmented_image.astype(np.uint8))
#         imageio.imwrite(updated_mask_path, updated_mask.astype(np.uint8))

#         print(f"Augmented image saved to {augmented_image_path}")
#         print(f"Updated mask saved to {updated_mask_path}")

#     def process_all_images(self):
#         src_files = [os.path.join(self.src_folder, f) for f in os.listdir(self.src_folder) if os.path.isfile(os.path.join(self.src_folder, f))]
#         overlay_files = {k: [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))] for k, (folder, _) in self.overlay_folders.items()}

#         for src_image_path in src_files:
#             overlay_image_paths = []
#             for category, files in overlay_files.items():
#                 count = self.overlay_folders[category][1]  # Number of overlays to select from this category
#                 overlay_image_paths.extend(random.sample(files, count))
#             print(f"Processing {src_image_path} with overlays {overlay_image_paths}")
#             self._process_image(src_image_path, overlay_image_paths)

# if __name__ == '__main__':
#     src_folder = '~/unet_v1.0/data_train_0/images_mini'
#     overlay_folders = {
#         'bubble': '~/unet_v1.0/artifacts/imgs/bubble',
#         'cell': '~/unet_v1.0/artifacts/imgs/cell',
#         'dark_spots': '~/unet_v1.0/artifacts/imgs/dark_spots',
#         'fat': '~/unet_v1.0/artifacts/imgs/fat',
#         'group': '~/unet_v1.0/artifacts/imgs/group',
#         'squamous': '~/unet_v1.0/artifacts/imgs/squamous',
#         'threads': '~/unet_v1.0/artifacts/imgs/threads'
#         # Add more categories and counts as needed
#     }
#     output_folder = '~/unet_v1.0/artifacts/output'

#     processor = ArtifactBatchProcessor(src_folder, overlay_folders, output_folder)
#     processor.process_all_images()








class ArtifactBatchProcessor:
    def __init__(self, src_folder, overlay_folders=None, output_folder=None):
        self.src_folder = os.path.expanduser(src_folder)

        # Define default overlay paths
        default_overlay_paths = {
            'bubble': '~/unet_v1.0/artifacts/imgs/bubble',
            'cell': '~/unet_v1.0/artifacts/imgs/cell',
            'dark_spots': '~/unet_v1.0/artifacts/imgs/dark_spots',
            'fat': '~/unet_v1.0/artifacts/imgs/fat',
            'group': '~/unet_v1.0/artifacts/imgs/group',
            'squamous': '~/unet_v1.0/artifacts/imgs/squamous',
            'threads': '~/unet_v1.0/artifacts/imgs/threads'
        }

        # If overlay_folders is None, initialize with default counts
        if overlay_folders is None:
            overlay_folders = {k: (v, 1) for k, v in default_overlay_paths.items()}

        # If overlay_folders is provided, merge with default paths
        else:
            for k, v in default_overlay_paths.items():
                if k in overlay_folders:
                    if isinstance(overlay_folders[k], tuple):
                        overlay_folders[k] = (v, overlay_folders[k][1])
                    else:
                        overlay_folders[k] = (v, overlay_folders[k])
                else:
                    overlay_folders[k] = (v, 1)

        self.overlay_folders = {k: (os.path.expanduser(v[0]), v[1]) for k, v in overlay_folders.items()}
        self.output_folder = os.path.expanduser(output_folder) if output_folder else os.path.join(self.src_folder, 'output')
        self.augmenter = ArtifactAugmentation()

        self._check_directories()

    def _check_directories(self):
        if not os.path.exists(self.src_folder) or not os.path.isdir(self.src_folder):
            raise FileNotFoundError(f"The source folder '{self.src_folder}' does not exist.")
        for folder, _ in self.overlay_folders.values():
            if not os.path.exists(folder) or not os.path.isdir(folder):
                raise FileNotFoundError(f"The overlay folder '{folder}' does not exist.")
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def _process_image(self, src_image_path, overlay_image_paths):
        src = imageio.imread(src_image_path)
        mask = np.ones((src.shape[0], src.shape[1]), dtype=np.float32)

        src_tensor = torch.from_numpy(src).permute(2, 0, 1).float() / 255.0  # (H, W, C) -> (C, H, W)
        mask_tensor = torch.from_numpy(mask).float()

        for overlay_image_path in overlay_image_paths:
            overlay = imageio.imread(overlay_image_path)
            overlay_mask = cv.cvtColor(overlay[:, :, :3], cv.COLOR_RGB2GRAY)
            overlay_mask = cv.resize(overlay_mask, (src.shape[1], src.shape[0]))

            augmented_image, updated_mask = self.augmenter.transparentOverlay(
                src=src_tensor,
                mask=mask_tensor,
                scale=0.5,
                mask_threshold=0.3,
                overlay_path=overlay_image_path,
            )

            src_tensor = augmented_image  # Update the src_tensor with the newly augmented image
            mask_tensor = updated_mask  # Update the mask_tensor with the updated mask

        augmented_image = augmented_image.permute(1, 2, 0).numpy() * 255.0
        updated_mask = updated_mask.numpy() * 255.0

        base_name = os.path.basename(src_image_path)
        augmented_image_path = os.path.join(self.output_folder, f"{base_name}")
        # updated_mask_path = os.path.join(self.output_folder, f"mask_{base_name}")

        imageio.imwrite(augmented_image_path, augmented_image.astype(np.uint8))
        # imageio.imwrite(updated_mask_path, updated_mask.astype(np.uint8))

        print(f"Augmented image saved to {augmented_image_path}")
        # print(f"Updated mask saved to {updated_mask_path}")

    def process_all_images(self):
        src_files = [os.path.join(self.src_folder, f) for f in os.listdir(self.src_folder) if os.path.isfile(os.path.join(self.src_folder, f))]
        overlay_files = {k: [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))] for k, (folder, _) in self.overlay_folders.items()}

        for src_image_path in src_files:
            overlay_image_paths = []
            for category, files in overlay_files.items():
                count = self.overlay_folders[category][1]  # Number of overlays to select from this category
                overlay_image_paths.extend(random.sample(files, count))
            print(f"Processing {src_image_path} with overlays {overlay_image_paths}")
            self._process_image(src_image_path, overlay_image_paths)



if __name__ == '__main__':
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

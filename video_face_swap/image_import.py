import os

PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = []
for i in range(1, 5):
    TEST_IMAGE_PATHS.append(os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)))

print(TEST_IMAGE_PATHS)
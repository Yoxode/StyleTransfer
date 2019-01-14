import urllib
import config
import os


def download_vgg(vgg_url=config.VGG_URL, vgg_path=config.VGG_PATH):
    if os.path.exists(vgg_path):
        print("VGG-19 pre-trained model ready")
        return
    print("Downloading the VGG pre-trained model. This might take a while ...")
    file_name, _ = urllib.request.urlretrieve(vgg_url, vgg_path)
    file_size = os.stat(file_name).st_size
    if file_size == 534904783:
        print('Successfully downloaded VGG-19 pre-trained model', file_name)
    else:
        raise Exception('File ' + file_name + ' might be corrupted. Try again later!.')


def make_dir(dir_path):
    if os.path.exists(dir_path):
        print("Directory " + dir_path + " already exists!")
    else:
        try:
            os.mkdir(dir_path)
        except OSError as e:
            print(e)
            pass

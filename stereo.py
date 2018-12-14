import time
import cv2
import pickle
import numpy as np
from numpy.linalg import inv
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch import Tensor
from matplotlib import pyplot as plt
from depthNet_model import depthNet
from visualize import *
import code


class Camera_Param(object):
    def __init__(self):
        self.images = []
        self.image_resize = []
        self.image_cuda = []
        self.CameraPose = []  # stores camera param for each images
        self.reference_index = -1  # stores the index of reference image
        self.indices = []  # stores the index of neighbor images

        # these paramets are taken from : https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
        # [0] stores left camera in world frame
        self.CameraPose.append((np.asarray([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0,  1., 0],
            [0,     0,   0, 1]])))

        # [1] stores right camera in world frame
        self.CameraPose.append((np.asarray([
            [9.9974346023742977e-01, -9.3241611192222736e-03,
                2.0641553524711167e-02, -9.3972656825422607e-02],
            [9.4704945013584906e-03,   9.9993063219576961e-01, -
                7.0028944350286094e-03, -8.6908678166320880e-04],
            [-2.0574825549454143e-02, 7.1965836333082178e-03,
                9.9976241464540883e-01, 2.3096349041841413e-03],
            [0,               0,              0,      1]])))

        self.camera_k = np.asarray([[4.6268606849104088e+02, 0., 3.1497954444748655e+02],  # intrinsic parameters
                                    [0, 4.6272665360184322e+02, 1.8753008193357198e+02],
                                    [0,     0,     1]])

        return

    def read(self, filenames, ref_idx):

        self.reference_index = ref_idx  # stores the index of the reference images
        self.indices = np.concatenate((np.array(range(0, ref_idx)),  # stores the index of the neighbor images
                                       np.array(range(ref_idx + 1, len(filenames)))), axis=0)
        # This reads the input images. For all images in filenames,
        images = [fn for fn in filenames]
        # This reads images.
        self.images = [cv2.imread(img) for img in images]
        return

    def epipolar_testing(self):
        # test the epipolar line
        ref_idx = self.reference_index

        for i in self.indices:  # epipolar testing for each neighbor image
            cur_idx = int(i)
            # Position of left cam wrt to right_cam
            left2right = np.dot(inv(self.CameraPose[cur_idx]), self.CameraPose[ref_idx])
            code.interact(local=locals())
            # Half of image size in X-Y axis
            test_point = np.asarray([self.images[ref_idx].shape[1] / 2,
                                     self.images[ref_idx].shape[0] / 2, 1])

            # Far point [X,Y,Z] wrt left camera.
            far_point = np.dot(inv(self.camera_k), test_point) * 50.0
            far_point = np.append(far_point, 1)                         # [X,Y,Z, 1]
            # Position of far point wrt to right camera
            far_point = np.dot(left2right, far_point)

            # Position of far pixel in right image
            far_pixel = np.dot(self.camera_k, far_point[0:3])
            far_pixel = (far_pixel / far_pixel[2])[0:2]                 # x = X/W, y = Y/W

            # near point [X,Y,Z] wrt camera.
            near_point = np.dot(inv(self.camera_k), test_point) * 0.01
            near_point = np.append(near_point, 1)                        # [X,Y,Z, 1]
            # Position of near point wrt to right camera
            near_point = np.dot(left2right, near_point)

            # Position of near pixel in right image
            near_pixel = np.dot(self.camera_k, near_point[0:3])
            near_pixel = (near_pixel / near_pixel[2])[0:2]               # x = X/W, y = Y/W

            cv2.line(self.images[cur_idx],
                     (int(far_pixel[0] + 0.5), int(far_pixel[1] + 0.5)),
                     (int(near_pixel[0] + 0.5), int(near_pixel[1] + 0.5)), [0, 0, 255], 4)
            cv2.circle(self.images[ref_idx], (int(test_point[0]),
                                              int(test_point[1])), 4, [0, 0, 255], -1)

            plt.subplot(121)
            plt.imshow(self.images[ref_idx])  # gray display
            plt.title('left')
            plt.subplot(122)
            plt.imshow(self.images[cur_idx])  # gray display
            plt.title('right')
            plt.show()

    def resize(self, new_width, new_height):
        # This resize the images
        self.image_resize = [cv2.resize(img, (new_width, new_height)) for img in self.images]
        return

    def pytorch_format(self):

        ref_idx = self.reference_index

        # changes the shape
        torch_left_image = np.moveaxis(self.image_resize[ref_idx], -1, 0)
        # changes the shape
        torch_left_image = np.expand_dims(torch_left_image, 0)
        torch_left_image = (torch_left_image - 81.0) / 35.0

        # process
        left_image_cuda = Tensor(torch_left_image).cuda()
        left_image_cuda = Variable(left_image_cuda, volatile=True)
        self.image_cuda.append(left_image_cuda)

        for i in self.indices:
            cur_idx = int(i)
            # changes the shape
            torch_right_image = np.moveaxis(self.image_resize[cur_idx], -1, 0)
            # changes the shape
            torch_right_image = np.expand_dims(torch_right_image, 0)
            torch_right_image = (torch_right_image - 81.0) / 35.0

            right_image_cuda = Tensor(torch_right_image).cuda()
            right_image_cuda = Variable(right_image_cuda, volatile=True)
            self.image_cuda.append(right_image_cuda)

        return self.image_cuda

    def camera_matrices(self, pixel_coordinate, factor_x, factor_y):

        KRKiUV_cuda_Ts = []
        KT_cuda_Ts = []
        self.camera_k[0, :] *= factor_x  # scaling the camera matrix
        self.camera_k[1, :] *= factor_y

        ref_idx = self.reference_index

        for i in self.indices:
            cur_idx = int(i)

            left2right = np.dot(inv(self.CameraPose[cur_idx]), self.CameraPose[0])
            left_in_right_T = left2right[0:3, 3]
            left_in_right_R = left2right[0:3, 0:3]
            K = self.camera_k
            K_inverse = inv(K)
            KRK_i = K.dot(left_in_right_R.dot(K_inverse))
            # projecting all pixels of reference_view to neighbor_view (equation 2). But only rotation considered
            KRKiUV = KRK_i.dot(pixel_coordinate)

            KT = K.dot(left_in_right_T)
            KT = np.expand_dims(KT, -1)  # changes the shape
            KT = np.expand_dims(KT, 0)  # changes the shape
            KT = KT.astype(np.float32)  # Assign the type format to float

            KRKiUV = KRKiUV.astype(np.float32)    # Assign the type format to float
            KRKiUV = np.expand_dims(KRKiUV, 0)    # changes the shape
            KRKiUV_cuda_T = Tensor(KRKiUV).cuda()
            KT_cuda_T = Tensor(KT).cuda()

            KRKiUV_cuda_Ts.append(KRKiUV_cuda_T)
            KT_cuda_Ts.append(KT_cuda_T)
        return KRKiUV_cuda_Ts, KT_cuda_Ts

    def estimate_depth(self, new_width, new_height):

        original_width = self.images[0].shape[1]
        original_height = self.images[0].shape[0]
        factor_x = new_width / original_width
        factor_y = new_height / original_height
        # Resizing the image
        self.resize(new_width, new_height)

        # convert to pythorch format (data shape changed)
        self.pytorch_format()

        # model
        depthnet = depthNet()
        model_data = torch.load('opensource_model.pth.tar')
        depthnet.load_state_dict(model_data['state_dict'])
        depthnet = depthnet.cuda()
        cudnn.benchmark = True
        depthnet.eval()

        # for warp the image to construct the cost volume
        pixel_coordinate = np.indices([new_width, new_height]).astype(
            np.float32)    # stores row and coloumn matrix
        pixel_coordinate = np.concatenate(
            (pixel_coordinate, np.ones([1, new_width, new_height])), axis=0)         # stores row, coloumn and third D matrix
        #  Convert individual matrix to a row vector
        pixel_coordinate = np.reshape(pixel_coordinate, [3, -1])

        KRKiUV_cuda_T, KT_cuda_T = self.camera_matrices(pixel_coordinate, factor_x, factor_y)

        depth = depthnet(self.image_cuda, KRKiUV_cuda_T, KT_cuda_T, new_width,
                         new_height, self.reference_index, self.indices)
        return depth


def plot_results(camera_param, depth):
    ref = camera_param.reference_index
    cur = camera_param.indices

    images = camera_param.images
    # visualize the results
    plt.subplot(251)
    plt.imshow(images[0])  # gray display
    plt.title('0')
    plt.subplot(252)
    plt.imshow(images[1])  # gray display
    plt.title('1')
    # plt.subplot(253)
    # plt.imshow(images[2])  # gray display
    # plt.title('2')
    # plt.subplot(254)
    # plt.imshow(images[3])  # gray display
    # plt.title('3')
    # plt.subplot(255)
    # plt.imshow(images[4])  # gray display
    # plt.title('4')

    plt.subplot(258)
    plt.imshow(depth[0], cmap='gray')  # gray display
    # plt.title('Ref:{}, N: ({:},{:},{:},{:})'.format(ref, cur[0], cur[1], cur[2], cur[3]))
    plt.show()


def main():

    # Pose and intrinsic parameters
    camera_param = Camera_Param()  # Fill in the camera param

    # Read images
    working_dir = "/media/sunil/DATADRIVE1/MyCode_GIT/DeepDepth/MultiView/MVS_Hkust/images/"
    filenames = [working_dir + '1.jpg', working_dir + '2.jpg']

    reference_index = 1     # this refers to image index for which depth is to be estimated

    camera_param.read(filenames, reference_index)

    # test the epipolar line
    camera_param.epipolar_testing()

    # Prediction
    new_width = 320
    new_height = 256
    depth = camera_param.estimate_depth(new_width, new_height)

    np_depth = []
    for i in range(len(camera_param.images) - 1):
        idepth = np.squeeze(depth[0][i].cpu().data.numpy())
        np_depth.append(np2Depth(idepth, np.zeros(idepth.shape, dtype=bool)))

    # visualize resutls
    plot_results(camera_param, np_depth)


main()

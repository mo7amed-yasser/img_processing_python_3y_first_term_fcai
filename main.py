import customtkinter as tk
import tkinter as tko
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import cv2 as ocv
import cv2 as cv2
from matplotlib import pyplot as plt
import numpy as np

windows=tk.CTk()
windows.geometry("500x100")
windows.title("imgprocessing")

windows.resizable(0,0)
val=tko.StringVar()

photopath=""

def imgpath():
    photopat = filedialog.askopenfilename(title="Select Image",
                                        filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg")))
    global photopath
    photopath = photopat


def comm(bb):
    ii=val.get()
    if photopath=="":
        messagebox.showerror("input error",message="Please choose an image from your device")
    else:
        img = ocv.imread(photopath, 0)
        ocv.imshow("before", img)
        if ii=="Normalize Image":
            Normalize_Image()
        if ii=="Negative Image":
            Negative_Image()
        if ii=="Power Law Image":
            Power_Law_Image()
        if ii=="Log Transformation Image":
            Log_Transformation_Image()
        if ii=="Max Filter":
            Max_Filter()
        if ii=="Min Filters":
            Min_Filters()
        if ii=="Single Threshold":
            Single_Threshold()
        if ii=="Multi Thresholding":
            Multi_Thresholding()
        if ii=="Contrast Stretching":
            Contrast_Stretching()
        if ii=="Histogram":
            Histogram()
        if ii=="Image Resize":
            Image_Resize()
        if ii=="Fourier Transformation":
            Fourier_Transformation()



def Normalize_Image():
    img = ocv.imread(photopath, 0)
    height, width = np.shape(img)
    img = img / 255
    ocv.imshow("Normalize Image", img)
    ocv.waitKey()
    ocv.destroyAllWindows()
def Negative_Image():
    img = ocv.imread(photopath, 0)
    height, width = np.shape(img)
    for row in range(height):
        for col in range(width):
            img[row][col] = 255 - img[row][col]
    ocv.imshow("My image", img)
    ocv.waitKey()
    ocv.destroyAllWindows()
def Power_Law_Image():
    img = ocv.imread(photopath, 0) / 255
    height, width = np.shape(img)
    for row in range(height):
        for col in range(width):
            img[row][col] = (2 * (img[row][col])) ** (3)
    ocv.imshow("Power Law", img)
    ocv.waitKey()
    ocv.destroyAllWindows()
def Log_Transformation_Image():
    img = ocv.imread(photopath, 0)
    height, width = np.shape(img)
    for row in range(height):
        for col in range(width):
            img[row][col] = 30 * (np.log2(img[row][col]))
    ocv.imshow("Log Transformation Image", img / 255)
    ocv.waitKey()
    ocv.destroyAllWindows()
def Max_Filter():
    img = ocv.imread(photopath, 0) / 255
    filter_val = [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]]
    width, height = np.shape(img)
    img2 = np.zeros((width, height))
    for row in range(1, width - 1):
        for col in range(1, height - 1):
            mat = [img[row][col], img[row][col + 1], img[row][col - 1], img[row - 1][col], img[row - 1][col + 1],
                   img[row - 1][col - 1], img[row + 1][col], img[row + 1][col + 1], img[row + 1][col - 1]]
            img2[row][col] = np.max(mat)
    ocv.imshow("After", img2)
    ocv.waitKey()
    ocv.destroyAllWindows()
def Min_Filters():
    im = ocv.imread(photopath, 0)
    row, col = np.shape(im)
    new = np.zeros((row, col))
    for r in range(2, row - 1, 1):
        for c in range(2, col - 1, 1):
            vect = np.array([im[r - 1][c + 1], im[r][c + 1], im[r + 1][c + 1], im[r - 1][c], im[r][c], im[r + 1][c],
                             im[r - 1][c - 1], im[r][c - 1], im[r + 1][c - 1]])
            new[r][c] = np.min(vect)
    max_val = np.max(new)
    min_val = np.min(new)
    new=(im.astype('float')-min_val)/(max_val-min_val)
    ocv.imshow("after", new)
    ocv.waitKey()
    ocv.destroyAllWindows()
def Single_Threshold():
    img = ocv.imread(photopath, 0) / 255
    width, height = np.shape(img)
    for row in range(width - 1):
        for col in range(height - 1):
            if img[row][col] < .7:
                img[row][col] = 0
            else:
                img[row][col] = 1
    ocv.imshow("After", img)
    ocv.waitKey()
    ocv.destroyAllWindows()
def Multi_Thresholding():
    img = ocv.imread(photopath, 0) / 255
    width, height = np.shape(img)
    for row in range(width - 1):
        for col in range(height - 1):
            if img[row][col] < .11:
                img[row][col] = 0
            elif img[row][col] > .11 and img[row][col] < .39:
                img[row][col] = .11
            elif img[row][col] > .39 and img[row][col] < .58:
                img[row][col] = .39
            else:
                img[row][col] = 1
    ocv.imshow("After", img)
    ocv.waitKey()
    ocv.destroyAllWindows()
def Contrast_Stretching():
    img = ocv.imread(photopath, 0) / 255
    width, height = np.shape(img)
    for row in range(width - 1):
        for col in range(height - 1):
            if (img[row][col] > .3) and (img[row][col] < .7):
                img[row][col] = img[row][col]
            else:
                img[row][col] = .2
    ocv.imshow("After", img)
    ocv.waitKey()
    ocv.destroyAllWindows()
def Histogram():
    img_array = ocv.imread(photopath, 0)
    histogram_array = np.bincount(img_array.flatten(), minlength=255)
    num_pixels = np.sum(histogram_array)
    histogram_array = histogram_array / num_pixels
    chistogram_array = np.cumsum(histogram_array)
    transform_map = np.floor(255 * chistogram_array).astype(np.uint8)
    img_list = list(img_array.flatten())
    eq_img_list = [transform_map[p] for p in img_list]
    eq_img_array = np.reshape(np.asarray(eq_img_list), img_array.shape)
    ocv.imshow("After", eq_img_array)
    ocv.waitKey()
    ocv.destroyAllWindows()
def Image_Resize():
    img = ocv.imread(photopath, 0) / 255
    width, height = np.shape(img)
    W = int(width / 2)
    H = int(height / 2)
    New_Img = []
    for row in range(1, width, 2):
        for col in range(1, height, 2):
            New_Img.append(img[row][col])
    New_Img = np.reshape(New_Img, (W, H))
    ocv.imshow("After", New_Img)
    ocv.waitKey()
    ocv.destroyAllWindows()
def Fourier_Transformation():
    image_path = photopath

    # read the input image
    image = cv2.imread(image_path, 0)

    # calculating the discrete Fourier transform
    DFT = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)

    # reposition the zero-frequency component to the spectrum's middle
    shift = np.fft.fftshift(DFT)
    row, col = image.shape
    center_row, center_col = row // 2, col // 2

    # create a mask with a centered square of 1s
    mask = np.zeros((row, col, 2), np.uint8)
    mask[center_row - 30:center_row + 30, center_col - 30:center_col + 30] = 1

    # put the mask and inverse DFT in place.
    fft_shift = shift * mask
    fft_ifft_shift = np.fft.ifftshift(fft_shift)
    imageThen = cv2.idft(fft_ifft_shift)

    # calculate the magnitude of the inverse DFT
    imageThen = cv2.magnitude(imageThen[:, :, 0], imageThen[:, :, 1])

    # visualize the original image and the magnitude spectrum
    plt.figure(figsize=(5, 5))
    plt.subplot(122), plt.imshow(imageThen, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()


header=tk.CTkFrame(windows,width=500,height=35,fg_color="steelblue")

title=tk.CTkFrame(windows,width=500,height=120)

messag=tk.CTkLabel(header,text="Choose the process you want to apply")


button=tk.CTkButton(windows,width=10,height=10,corner_radius=10,text="image",command=lambda :imgpath())



processing=("Normalize Image","Negative Image","Power Law Image","Log Transformation Image","Max Filter","Min Filters","Single Threshold","Multi Thresholding","Contrast Stretching","Histogram","Image Resize","Fourier Transformation")



optionmenu=tko.ttk.OptionMenu(header,val,processing[0],*processing,command=comm)
optionmenu.place(x=10,y=7)



messag.place(x=200,y=2)

button.place(x=40,y=50)

header.place(x=0,y=0)

title.place(x=0,y=30)


windows.mainloop()


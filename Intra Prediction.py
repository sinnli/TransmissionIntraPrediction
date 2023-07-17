
import numpy as np
import cv2

def predict_sad(left_neighbor,upper_left_neighbor,upper_neighbor,upper_right_neighbor,current):

    sads = []
    predicted_blocks = []
    modes =["DC","vertical","horizontal","diagonal down left",
            "diagonal down right","veritcal left",
            "veritcal right","horizontal down",
            "horizontal up"]

    #DC prediction
    gr = (np.sum(left_neighbor[:,3])+np.sum(upper_neighbor[3,:])+4)//8
    DC_block = np.full((4,4),gr)
    predicted_blocks.append(DC_block)
    DC_sad = np.sum(np.abs(current - DC_block))
    sads.append(DC_sad)

    #vertical prediction
    gr = upper_neighbor[3,0]
    bl = upper_neighbor[3,1]
    pi = upper_neighbor[3,2]
    ye = upper_neighbor[3,2]
    Vr_block = np.zeros((4,4))
    Vr_block[:,0] = gr
    Vr_block[:, 1] = bl
    Vr_block[:, 2] = pi
    Vr_block[:, 3] = ye
    Vr_block = Vr_block.astype(int)
    predicted_blocks.append(Vr_block)
    VR_sad = np.sum(np.abs(current-Vr_block))
    sads.append(VR_sad)

    #horizontal prediction
    gr = left_neighbor[0,3]
    bl = left_neighbor[1,3]
    pi = left_neighbor[2,3]
    ye = left_neighbor[3,3]
    Hr_block = np.zeros((4,4))
    Hr_block[0,:] = gr
    Hr_block[1,:] = bl
    Hr_block[2,:] = pi
    Hr_block[3,:] = ye
    Hr_block = Hr_block.astype(int)
    predicted_blocks.append(Hr_block)
    HR_sad = np.sum(np.abs(current-Hr_block))
    sads.append(HR_sad)

    #diagonal down left prediction
    gr = (upper_neighbor[3,0]+2*upper_neighbor[3,1]+upper_neighbor[3,2]+2)//4
    bli = (upper_neighbor[3,1]+2*upper_neighbor[3,2]+upper_neighbor[3,3]+2)//4
    pi = (upper_neighbor[3,2]+2*upper_neighbor[3,3]+upper_right_neighbor[3,0]+2)//4
    ye = (upper_neighbor[3,3]+2*upper_right_neighbor[3,0]+upper_right_neighbor[3,1]+2)//4
    pu = (upper_right_neighbor[3,0]+2*upper_right_neighbor[3,1]+upper_right_neighbor[3,2]+2)//4
    bch = (upper_right_neighbor[3,1]+2*upper_right_neighbor[3,2]+upper_right_neighbor[3,3]+2)//4
    bdk =(upper_right_neighbor[3,2]+3*upper_right_neighbor[3,3]+2)//4
    Dl_block = np.zeros((4,4))
    Dl_block[0,0] = gr
    Dl_block[0,1] = Dl_block[1,0] = bli
    Dl_block[0,2] = Dl_block[1,1] = Dl_block[2,0] = pi
    Dl_block[0,3] = Dl_block[1,2] = Dl_block[2,1] = Dl_block[3,0] = ye
    Dl_block[1,3] = Dl_block[2, 2] = Dl_block[3, 1] = pu
    Dl_block[2,3] = Dl_block[3, 2] = bch
    Dl_block[3,3] = bdk
    Dl_block = Dl_block.astype(int)
    predicted_blocks.append(Dl_block)
    Dl_sad = np.sum(np.abs(current-Dl_block))
    sads.append(Dl_sad)


    #diagonal down right prediction
    gr = (left_neighbor[3,3]+2*left_neighbor[2,3]+left_neighbor[1,3]+2)//4
    bli = (left_neighbor[2,3]+2*left_neighbor[1,3]+left_neighbor[0,3]+2)//4
    pi = (left_neighbor[1,3]+2*left_neighbor[0,3]+upper_left_neighbor[3,3]+2)//4
    ye = (left_neighbor[0,3]+2*upper_left_neighbor[3,3]+upper_neighbor[3,0]+2)//4
    pu = (upper_left_neighbor[3,3]+2*upper_neighbor[3,0]+upper_neighbor[3,1]+2)//4
    bch = (upper_neighbor[3,0]+2*upper_neighbor[3,1]+upper_neighbor[3,3]+2)//4
    bdk =(upper_neighbor[3,1]+2*upper_neighbor[3,2]+upper_neighbor[3,3]+2)//4
    Dr_block = np.zeros((4,4))
    Dr_block[3,0] = gr
    Dr_block[2,0] = Dr_block[3,1] = bli
    Dr_block[1,0] = Dr_block[2,1] = Dr_block[3,2] = pi
    Dr_block[0,0] = Dr_block[1,1] = Dr_block[2,2] = Dr_block[3,3] = ye
    Dr_block[0,1] = Dr_block[1, 2] = Dr_block[2, 3] = pu
    Dr_block[0,2] = Dr_block[1, 3] = bch
    Dr_block[0,3] = bdk
    Dr_block = Dr_block.astype(int)
    predicted_blocks.append(Dr_block)
    Dr_sad = np.sum(np.abs(current-Dr_block))
    sads.append(Dr_sad)


    #veritcal left prediction
    gr = (upper_neighbor[3,0]+upper_neighbor[3,1]+1)//2
    bli = (upper_neighbor[3,1]+upper_neighbor[3,2]+1)//2
    pi = (upper_neighbor[3,2]+upper_neighbor[3,3]+1)//2
    ye = (upper_neighbor[3,3]+upper_right_neighbor[3,0]+1)//2
    pu = (upper_right_neighbor[3,0]+upper_right_neighbor[3,1]+1)//2
    bch = (upper_neighbor[3,0]+2*upper_neighbor[3,1]+upper_neighbor[3,2]+2)//4
    bdk = (upper_neighbor[3,1]+2*upper_neighbor[3,2]+upper_neighbor[3,3]+2)//4
    gry = (upper_neighbor[3,2]+2*upper_neighbor[3,3]+upper_right_neighbor[3,0]+2)//4
    turkis = (upper_neighbor[3,3]+2*upper_right_neighbor[3,0]+upper_right_neighbor[3,1]+2)//4
    white = (upper_right_neighbor[3,0]+2*upper_right_neighbor[3,1]+upper_right_neighbor[3,2]+2)//4
    Vrl_block = np.zeros((4,4))
    Vrl_block[0,0] = gr
    Vrl_block[0,1] = Vrl_block[2,0] = bli
    Vrl_block[0,2] = Vrl_block[2,1] = pi
    Vrl_block[0,3] = Vrl_block[2,2] = ye
    Vrl_block[2,3] = pu
    Vrl_block[1,0] = bch
    Vrl_block[1,1] = Vrl_block[3,0] = bdk
    Vrl_block[1,2] = Vrl_block[3,1] = gry
    Vrl_block[1,3] = Vrl_block[3,2] = turkis
    Vrl_block[3,3] = white
    Vrl_block = Vrl_block.astype(int)
    predicted_blocks.append(Vrl_block)
    Vrl_sad = np.sum(np.abs(current-Vrl_block))
    sads.append(Vrl_sad)


    # veritcal right prediction
    gr = (upper_left_neighbor[3, 3] + upper_neighbor[3, 0] + 1) // 2
    bli = (upper_neighbor[3, 0] + upper_neighbor[3, 1] + 1) // 2
    pi = (upper_neighbor[3, 1] + upper_neighbor[3, 2] + 1) // 2
    ye = (upper_neighbor[3, 2] + upper_neighbor[3, 3] + 1) // 2
    pu = (left_neighbor[0, 3] + 2*upper_left_neighbor[3, 3]+upper_neighbor[3,0] + 2) // 4
    bch = (upper_left_neighbor[3, 3] + 2*upper_neighbor[3, 0]+upper_neighbor[3,1] + 2) // 4
    bdk = (upper_neighbor[3, 0] + 2*upper_neighbor[3, 1]+upper_neighbor[3,2] + 2) // 4
    gry = (upper_neighbor[3, 1] + 2*upper_neighbor[3, 2]+upper_neighbor[3,3] + 2) // 4
    turkis = (upper_left_neighbor[3, 3] + 2 * left_neighbor[0, 3] + left_neighbor[1, 3] + 2) // 4
    white = (left_neighbor[0, 3] + 2 * left_neighbor[1, 3] + left_neighbor[2, 3] + 2) // 4
    Vrr_block = np.zeros((4, 4))
    Vrr_block[0, 0] = Vrr_block[2,1] = gr
    Vrr_block[0, 1] = Vrr_block[2, 2] = bli
    Vrr_block[0, 2] = Vrr_block[2, 3] = pi
    Vrr_block[0, 3] = ye
    Vrr_block[1, 0] = Vrr_block[3,1] = pu
    Vrr_block[1, 1] = Vrr_block[3,2] = bch
    Vrr_block[1, 2] = Vrr_block[3, 3] = bdk
    Vrr_block[1, 3] = gry
    Vrr_block[2, 0] = turkis
    Vrr_block[3, 0] = white
    Vrr_block = Vrr_block.astype(int)
    predicted_blocks.append(Vrr_block)
    Vrr_sad = np.sum(np.abs(current - Vrr_block))
    sads.append(Vrr_sad)



    # horizontal down prediction
    gr = (upper_left_neighbor[3, 3] + left_neighbor[0, 3] + 1) // 2
    bli = (left_neighbor[0, 3] +2*upper_left_neighbor[3, 3]+upper_neighbor[3,0] + 2) // 4
    pi = (upper_left_neighbor[3, 3] +2*upper_neighbor[3, 0]+upper_neighbor[3,1] + 2) // 4
    ye = (upper_neighbor[3, 0] +2*upper_neighbor[3, 1]+upper_neighbor[3,2] + 2) // 4
    pu = np.float32((left_neighbor[0, 3] + left_neighbor[1, 3] + 1) / 2)
    bch = (upper_left_neighbor[3, 3] + 2*left_neighbor[0, 3]+left_neighbor[1,3] + 2) // 4
    bdk = (upper_left_neighbor[1, 3] + left_neighbor[2, 3] + 1) // 2
    gry = (left_neighbor[0, 3] + 2*left_neighbor[1, 3]+left_neighbor[2,3] + 2) // 4
    turkis = np.float32((left_neighbor[2, 3] + left_neighbor[3, 3] + 1) / 2)
    white = (left_neighbor[1, 3] + 2 * left_neighbor[2, 3] + left_neighbor[3, 3] + 2) // 4
    Hrd_block = np.zeros((4, 4))
    Hrd_block[0, 0] = Hrd_block[1,2] = gr
    Hrd_block[0, 1] = Hrd_block[1, 3] = bli
    Hrd_block[0, 2] = pi
    Hrd_block[0, 3] = ye
    Hrd_block[1, 0] = Hrd_block[2,2] = pu
    Hrd_block[1, 1] = Hrd_block[2,3] = bch
    Hrd_block[2, 0] = Hrd_block[3, 2] = bdk
    Hrd_block[2, 1] = Hrd_block[3,3]= gry
    Hrd_block[3, 0] = turkis
    Hrd_block[3, 1] = white
    Hrd_block = Hrd_block.astype(int)
    predicted_blocks.append(Hrd_block)
    Hrd_sad = np.sum(np.abs(current - Hrd_block))
    sads.append(Hrd_sad)



    # horizontal up prediction
    gr = np.float32((left_neighbor[0, 3] + left_neighbor[1, 3] + 1) / 2)
    bli = (left_neighbor[0, 3] +2*left_neighbor[1, 3]+left_neighbor[2,3] + 2) // 4
    pi = np.float32((left_neighbor[1, 3] + left_neighbor[2, 3] + 1) / 2)
    ye = (left_neighbor[1, 3] +2*left_neighbor[2, 3]+left_neighbor[3,3] + 2) // 4
    pu = np.float32((left_neighbor[2, 3] + left_neighbor[3, 3] + 1) / 2)
    bch = (left_neighbor[2, 3] + 2*left_neighbor[3, 3]+left_neighbor[3,3] + 2) // 4
    bdk = left_neighbor[3,3]
    Hru_block = np.zeros((4, 4))
    Hru_block[0, 0] = gr
    Hru_block[0, 1] = bli
    Hru_block[0, 2] = Hru_block[1,0] = pi
    Hru_block[0, 3] = Hru_block[1,1] = ye
    Hru_block[1, 2]= Hru_block[2,0] = pu
    Hru_block[1, 3] = Hru_block[2,1] = bch
    Hru_block[2, 2] = Hru_block[2,3] = Hru_block[3,0]= Hru_block[3,1]= Hru_block[3,2]= Hru_block[3,3] = bdk
    Hru_block = Hru_block.astype(int)
    predicted_blocks.append(Hru_block)
    Hru_sad = np.sum(np.abs(current - Hru_block))
    sads.append(Hru_sad)



    return modes[np.argmin(sads)], predicted_blocks[np.argmin(sads)]


def DCT(residual_block):
    mat1 = np.array([[1,1,1,1],[2,1,-1,2],[1,-1,-1,1],[1,-2,2,-1]])
    mat2 = np.array([[1,2,1,1],[1,1,-1,-2],[1,-1,-1,2],[1,-2,1,-1]])
    return np.dot(mat1, np.dot(residual_block, mat2))

def Quantization(Y,QP):
    mat = np.array([[13107,8066,13107,8066],[8066,5243,8066,5243],[13107,8066,13107,8066],[8066,5243,8066,5243]])
    num = 1/(2**(15+(QP/6)))
    return np.round(Y*mat*num)

def IDCT(Y,QP):
    mat1 = np.array([[1,1,1,0.5],[1,0.5,-1,-1],[1,-0.5,-1,1],[1,-1,1,-0.5]])
    mat2 = np.array([[10,13,10,13],[13,16,13,16],[10,13,10,13],[13,16,13,16]])
    num = 2**(QP/6)
    mat3 = np.array([[1,1,1,1],[1,0.5,-0.5,-1],[1,-1,-1,1],[0.5,-1,1,-0.5]])
    temp = Y*mat2*num
    return np.round(np.dot(np.dot(mat1,temp),mat3)* (1/(2**6)))

def numSequnces(array):
    previous = array[0]
    flag = 1
    numSeq = 0

    for num in array:
        if (num-previous)==1:
            if(flag):
                flag =0
                numSeq+=1
        elif (flag ==0):
            flag = 1
        previous = num

    return numSeq



if __name__ == "__main__":
    QP = 12  #6, 12, 18 , 30
    image = np.load('flowers.npy')
    #Starting with 4x4 Luma Prediction
    # Assuming you have the image stored as an npy array
    modes = [] #question 3
    # Get the dimensions of the image
    image_height, image_width = image.shape

    reconstruted_im = image.copy()
    residue_im = np.zeros((image_height,image_width)).astype(int)
    inter_prd_im = np.zeros((image_height,image_width)).astype(int)

    # Iterate over each 4x4 block within the image
    for i in range(image_height // 4):
        for j in range(image_width // 4):
            # Extract the current 4x4 block
            current_block = reconstruted_im[i*4:(i+1)*4, j*4:(j+1)*4]
            # Get the left neighbor block
            if j > 0:
                left_neighbor = reconstruted_im[i*4:(i+1)*4, (j-1)*4:j*4]
            else:
                left_neighbor = np.full((4, 4), 128)  # Left neighbor block does not exist, assign 128
            # Get the upper left neighbor block
            if i > 0 and j > 0:
                upper_left_neighbor = reconstruted_im[(i-1)*4:i*4, (j-1)*4:j*4]
            else:
                upper_left_neighbor = np.full((4, 4), 128)  # Upper left neighbor block does not exist, assign 128

            # Get the block above the current block
            if i > 0:
                upper_neighbor = reconstruted_im[(i-1)*4:i*4, j*4:(j+1)*4]
            else:
                upper_neighbor = np.full((4, 4), 128)  # Block above the current block does not exist, assign 128

            # Get the upper right neighbor block
            if i > 0 and j < image_width // 4 - 1:
                upper_right_neighbor = reconstruted_im[(i-1)*4:i*4, (j+1)*4:(j+2)*4]
            else:
                upper_right_neighbor = np.full((4, 4), 128)  # Upper right neighbor block does not exist, assign 128

            #check the predicted best option and update
            mode ,predicted_block = predict_sad(left_neighbor, upper_left_neighbor, upper_neighbor, upper_right_neighbor, current_block)
            modes.append(mode) #question 3
            inter_prd_im[i * 4:(i + 1) * 4, j * 4:(j + 1) * 4]= predicted_block
            original_block = image[i * 4:(i + 1) * 4, j * 4:(j + 1) * 4]
            residue_block = original_block-predicted_block
            residue_im[i * 4:(i + 1) * 4, j * 4:(j + 1) * 4] = residue_block

            #Transforming
            Y = DCT(residue_block)

            #Quntization
            Y = Quantization(Y,QP)

            #Inverse transform
            Z = IDCT(Y.astype(int),QP)

            # reconstructed block = predicted block + residual block
            current_block = np.round(predicted_block+Z).astype(int)

            #update predictions for further blocks
            reconstruted_im[i * 4:(i + 1) * 4, j * 4:(j + 1) * 4]= current_block



    # Display the image using cv2.imshow()
    cv2.imshow('Original', image.astype(np.uint8))
    cv2.waitKey(0)
    cv2.imshow('Inter Predictions', inter_prd_im.astype(np.uint8))
    cv2.waitKey(0)
    cv2.imshow('Residue', residue_im.astype(np.uint8))
    cv2.waitKey(0)
    cv2.imshow('Reconstructed image', reconstruted_im.astype(np.uint8))
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    #question 3 - a
    unique_elements, counts = np.unique(modes, return_counts=True)
    # Print the incidence for each element
    for element, count in zip(unique_elements, counts):
        print(f"{element}: {count}")


    # #quesiton 3 - b
    # #first getting the indxes of each of the modes used in the image
    # modes_name = ['DC', 'vertical', "horizontal", "diagonal down left",
    #          "diagonal down right", "veritcal left",
    #          "veritcal right", "horizontal down",
    #          "horizontal up"]
    #
    # DC =[]
    # vertical = []
    # horizontal = []
    # diagonal_d_l = []
    # diagonal_d_r = []
    # vertical_l = []
    # vertical_r = []
    # horizontal_d = []
    # horizontal_u = []
    #
    #
    # for inx in range(len(modes)):
    #     if (modes[inx]=='DC'):
    #         DC.append(inx)
    #     elif (modes[inx]=='vertical'):
    #         vertical.append(inx)
    #     elif (modes[inx] == 'horizontal'):
    #         horizontal.append(inx)
    #     elif (modes[inx] == 'diagonal down left'):
    #         diagonal_d_l.append(inx)
    #     elif (modes[inx] == 'diagonal down right'):
    #         diagonal_d_r.append(inx)
    #     elif (modes[inx] == 'veritcal left'):
    #         vertical_l.append(inx)
    #     elif (modes[inx] == 'veritcal right'):
    #         vertical_r.append(inx)
    #     elif (modes[inx] == 'horizontal down'):
    #        horizontal_d.append(inx)
    #     elif (modes[inx] == 'horizontal up'):
    #        horizontal_u.append(inx)
    #
    # #know getting number of seqences
    # print("Sequences for each mode:")
    # print(f"DC: {numSequnces(DC)}")
    # print(f"vertical: {numSequnces(vertical)}")
    # print(f"horizontal: {numSequnces(horizontal)}")
    # print(f"diagonal down left: {numSequnces(diagonal_d_l)}")
    # print(f"diagonal down right: {numSequnces(diagonal_d_r)}")
    # print(f"veritcal left: {numSequnces(vertical_l)}")
    # print(f"veritcal right: {numSequnces(vertical_r)}")
    # print(f"horizontal down: {numSequnces(horizontal_d)}")
    # print(f"horizontal up: {numSequnces(horizontal_u)}")
































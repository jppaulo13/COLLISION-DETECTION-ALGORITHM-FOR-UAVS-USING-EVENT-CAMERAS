import sys, os, shutil
import yaml
import cv2
import numpy as np
from math import fabs, sqrt
import pandas as pd
import pyarrow
import matplotlib.colors as colors
from PIL import Image as im
import math
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from PIL import Image as im
from memory_profiler import profile
import timeit

with_rosbag = False
try:
    import rosbag
except:
    with_rosbag = False

# The dvs-related functionality implemented in C.
import cpydvs
import myModule

class bcolors:
    HEADER = '\033[95m'
    PLAIN = '\033[37m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def offset(str_, p_offset):
    for i in range(p_offset):
        str_ = '...' + str_
    return str_

def hdr(str_, p_offset=0):
    return offset(bcolors.HEADER + str_ + bcolors.ENDC, p_offset)

def wht(str_, p_offset=0):
    return offset(bcolors.PLAIN + str_ + bcolors.ENDC, p_offset)

def okb(str_, p_offset=0):
    return offset(bcolors.OKBLUE + str_ + bcolors.ENDC, p_offset)

def okg(str_, p_offset=0):
    return offset(bcolors.OKGREEN + str_ + bcolors.ENDC, p_offset)

def wrn(str_, p_offset=0):
    return offset(bcolors.WARNING + str_ + bcolors.ENDC, p_offset)

def err(str_, p_offset=0):
    return offset(bcolors.FAIL + str_ + bcolors.ENDC, p_offset)

def bld(str_, p_offset=0):
    return offset(bcolors.BOLD + str_ + bcolors.ENDC, p_offset)


# legacy is for EVIMO1, generation is very slow (ten minutes or more)
# using binary search is much faster (takes maybe 1 second) but gives
# slightly different indices having the exact indices is not a problem
# for the purpose of the discretization
#@profile
def get_index(cloud, index_w, legacy=False):
    #print (okb("Indexing..."))

    idx = [0]
    if (cloud.shape[0] < 2):
        return np.array(idx, dtype=np.uint32)

    if not legacy:
        index_times = np.arange(cloud[0, 0], cloud[-1, 0], step=index_w)#np.arange starts at cloud[0,0] stops at cloud[-1,0] (not couting the last value) by step  
        idx = np.searchsorted(cloud[:, 0], index_times, side='left')#gives the positions where index times would be inserted in cloud to maintain order
        idx = np.concatenate((idx, (cloud.shape[0]-1,)))
        idx = idx.astype(np.uint32)
    else:
        last_ts = cloud[0][0]
        for i, e in enumerate(cloud):
            if i % 100000 == 0:
                sys.stdout.write("\r" + str(i + 1) + ' / ' +str(len(cloud)) + '\t\t')
            while (e[0] - last_ts > index_w):
                if (e[0] - last_ts > 1.0):
                    print (wrn("\nGap in the events:"), e[0] - last_ts, 'sec.')
                idx.append(i)
                last_ts += index_w

        idx.append(cloud.shape[0] - 1)
        idx = np.array(idx, dtype=np.uint32)
    return idx

#@profile
def find_closest_idx(arr, val):
       idx = np.abs(arr - val).argmin()
       return idx

#@profile
def read_event_file_txt(fname, discretization, sort=False, legacy_discretization=False):
    #print (okb("Reading the event file as a text file..."))
    # For a 140M event file (~25 seconds), pandas takes 60 seconds with the C engine
    # with pyarrow it takes 23 seconds, pyarrow is considered experimental at the time of writing
    # np.loadtxt takes 765 seconds
    if os.path.exists(fname):  #checks if specific path exists or not
        try:
            cloud_pd = pd.read_csv(fname,                      #read a cvs file
                                    dtype=np.float64,           #type of the values
                                    names=['t', 'x', 'y', 'p'], #name of the columns
                                    delimiter=' ',
                                    engine='pyarrow').to_numpy() #fast libraries to deal to data
            # cloud_pd = pd.read_csv(fname,                       #read a cvs file
            #                         dtype=np.float64,           #type of the values
            #                         names=['x', 'y', 'p','t'], #name of the columns
            #                         delimiter=',',
            #                         engine='pyarrow',
            #                         ).to_numpy() #fast libraries to deal to data
    
            # Something about what panda's to_numpy returns breaks codes that follow
            # copying the pandas data into another numpy array fixes it, very strange
            cloud = np.zeros(cloud_pd.shape, dtype=np.float32)
            cloud[:, :] = cloud_pd[:, :]
            # All these things are identical.....
            #print(cloud_pd.shape)
            #print(cloud_pd.dtype)
            #print(cloud.shape)
            #print(cloud.dtype)
            #print(type(cloud_pd))
            #print(type(cloud))
        except pyarrow.lib.ArrowInvalid:
           
            # CSV was empty because this is probably a conventional camera sequence
            cloud = np.zeros((0,), dtype=np.float32)
    else:
        cloud = np.zeros((0,), dtype=np.float32)        
    
    if (sort):
        cloud = cloud[cloud[:,0].argsort()]
        
    if (cloud.shape[0] == 0):                       
        #print (wrn("Read 0 events from " + fname + "!"))
        return np.empty(shape=(0,4), dtype=np.float32), np.empty(shape=(0,), dtype=np.float32) 
    else:
        cloud[:,0] = cloud[:,0]*0.000001
        t0 = cloud[0][0]
        cloud[:,0] -= t0
        #print (wrn("Adjusting initial timestamp to 0 and from miliseconds to seconds!"))

    #print (okg("Read"), cloud.shape[0], okg("events:"), cloud[0][0], "-", cloud[-1][0], "sec.")

    idx = get_index(cloud, discretization, legacy_discretization)
    return cloud.astype(np.float32), idx

#@profile
def read_imu_file(fname_imu,ti,tf): 
#reads imu data and stores in cloud_imu but only between the timestmap ti and tf
    #print (okb("Reading the imu file as a text file..."))
    if os.path.exists(fname_imu):  #checks if specific path exists or not
        try:
            cloud_imu_pd = pd.read_csv(fname_imu,                      #read a cvs file
                                    dtype=np.float64,           #type of the values
                                    names=['t', 'gx', 'gy', 'gz'], #name of the columns
                                    delimiter=' ',
                                    engine='pyarrow').to_numpy() #fast libraries to deal to data

            cloud_imu = np.zeros(cloud_imu_pd.shape, dtype=np.float32)
            cloud_imu[:, :] = cloud_imu_pd[:, :]

        except pyarrow.lib.ArrowInvalid:
           
            # CSV was empty because this is probably a conventional camera sequence
            cloud_imu = np.zeros((0,), dtype=np.float32)
    else:
        cloud_imu = np.zeros((0,), dtype=np.float32)        
        
    if (cloud_imu.shape[0] == 0):                       
        #print (wrn("Read 0 events from " + fname_imu + "!"))
        return np.empty(shape=(0,4), dtype=np.float32), np.empty(shape=(0,), dtype=np.float32) 
    else:
        cloud_imu[:,0] = cloud_imu[:,0]*0.000001
        t0 = cloud_imu[0][0]
        cloud_imu[:,0] -= t0
        #print (wrn("Adjusting initial timestamp to 0 and from miliseconds to seconds!"))
    
    idx_i = find_closest_idx(cloud_imu[:,0],ti)
    idx_f = find_closest_idx(cloud_imu[:,0],tf)
    
    sl_cloud_imu = np.copy(cloud_imu[idx_i:idx_f].astype(np.float32))

    return sl_cloud_imu.astype(np.float32)

#@profile
def get_slice(cloud, idx, ts, width, mode=0, idx_step=0.01):
    if (cloud.shape[0] == 0):
        return cloud, np.array([0])

    ts_lo = ts
    ts_hi = ts + width
    if (mode == 1):
        ts_lo = ts - width / 2.0
        ts_hi = ts + width / 2.0
    if (mode == 2):
        ts_lo = ts - width
        ts_hi = ts
    if (mode > 2 or mode < 0):
        print (wrn("get_slice: Wrong mode! Reverting to default..."))
    if (ts_lo < 0): ts_lo = 0

    t0 = cloud[0][0]

    idx_lo = int((ts_lo - t0) / idx_step)
    idx_hi = int((ts_hi - t0) / idx_step)
    if (idx_lo >= len(idx)): idx_lo = -1
    if (idx_hi >= len(idx)): idx_hi = -1

    sl = np.copy(cloud[idx[idx_lo]:idx[idx_hi]].astype(np.float32)) #creates copied array from idx_low to idx_high
    idx_ = np.copy(idx[idx_lo:idx_hi])

    if (idx_lo == idx_hi):
        return sl, np.array([0])

    if (len(idx_) > 0):
        idx_0 = idx_[0]
        idx_ -= idx_0

    if (sl.shape[0] > 0):
        t0 = sl[0][0]
        sl[:,0] -= t0

    return sl, idx_

#@profile
def cut_negative_events(sl): #deletes all events which are negative
    idx_negatives = []
    for i in range (0, sl.shape[0]):
        if(int(sl[i,3]) == 0):
            idx_negatives = np.append (idx_negatives,i)
    idx_negatives = idx_negatives.astype(int)     
    sl = np.delete(sl,idx_negatives,axis=0)
    return sl

#@profile
def show_frame(sl,x,y):    
# x is the height of the image 
# y is the width of the image
#Show frame with blue positive events e red negative events.  
    shape = np.zeros((x,y,3))
    frame = np.zeros((4 * sl.shape[0]), dtype=np.float64)
    frame = sl.flatten()
    for i in range (0, sl.shape[0]):
        if(frame[(i * 4)+3] == 1):
            shape[int(frame[(i * 4)+2]),int(frame[(i * 4)+1]),0] += 1
        else:
            shape[int(frame[(i * 4)+2]),int(frame[(i * 4)+1]),0] -= 1
                
        
    for i in range (0,x):           #HSV
        for j in range (0,y):
            if(shape[i,j,0])>0:
                shape[i,j,0] = np.uint8('500')/255 #BLUE
                shape[i,j,1] = np.uint8('255')/255
                shape[i,j,2] = np.uint8('255')/255
            elif(shape[i,j,0])<0:
                shape[i,j,0] = np.uint8('180')/255 #RED
                shape[i,j,1] = np.uint8('255')/255
                shape[i,j,2] = np.uint8('255')/255
            else:
                shape[i,j,0] = np.uint8('458')/255 #BLACK
                shape[i,j,1] = np.uint8('458')/255
                shape[i,j,2] = np.uint8('0')/255
                
    img = colors.hsv_to_rgb(shape)
    return img
    
#@profile
def average_imu(cloud_imu):
    gx = np.mean(cloud_imu[:,1])
    if (math.isnan(gx)):
        gx = 1
    gy = np.mean(cloud_imu[:,2])
    if (math.isnan(gy)):
        gy = 1
    gz = np.mean(cloud_imu[:,3])
    if (math.isnan(gz)):
        gz = 1
    vector_g = np.array([gx,gy,gz],dtype = np.float64)
    return vector_g

#@profile
def rodrigues(vector_g):
    rodrigues_matrix = np.zeros([3,3])
    cv2.Rodrigues(vector_g,rodrigues_matrix)
    return rodrigues_matrix

#@profile
def ego_motion(sl,height,width,cloud_imu):  
    vector_w = average_imu(cloud_imu) #get the average of the imu data
    matrix_rot = rodrigues(vector_w)  #get the rotation matrix through the rodrigues algorithm
    a00 = matrix_rot[0,0]
    a01 = matrix_rot[0,1]
    a02 = matrix_rot[0,2]
    a10 = matrix_rot[1,0]
    a11 = matrix_rot[1,1]
    a12 = matrix_rot[1,2]
    a20 = matrix_rot[2,0]
    a21 = matrix_rot[2,1]
    a22 = matrix_rot[2,2]
    
    warp_image = np.zeros((height,width,5),dtype=np.float64)
    t0 = sl[0,0]; #save first timestamp
    cont = 0
    total_time = 0
    for i in range(0,sl.shape[0]):    
        x = sl[i,2]
        y = sl[i,1]
        t = sl[i,0]
        x_linha = round((a00 * x) + (a01 * y) + (a02 * t))
        y_linha = round((a10 * x) + (a11 * y) + (a12 * t))
        
        if (x_linha>=0 and x_linha<height and y_linha>=0 and y_linha<width): #between bounds 
            warp_image[x_linha,y_linha,0] = 1; #pixels that have atleast one event
            warp_image[x_linha,y_linha,1] +=1; #sum of events in pixel
            warp_image[x_linha,y_linha,2] +=t; #sum of the timestamps from events in pixel
            cont+=1
            total_time+= t
        # elif (x_linha<0 and y_linha>=0 and y_linha<width)  :
        #     warp_image[0,y_linha,0] = 1; #pixels that have atleast one event
        #     warp_image[0,y_linha,1] +=1; #sum of events in pixel
        #     warp_image[0,y_linha,2] +=t; #sum of the timestamps from events in pixel
        #     cont+=1
        #     total_time+= t
        # elif (x_linha<0 and y_linha<0)  :
        #     warp_image[0,0,0] = 1; #pixels that have atleast one event
        #     warp_image[0,0,1] +=1; #sum of events in pixel
        #     warp_image[0,0,2] +=t; #sum of the timestamps from events in pixel
        #     cont+=1
        #     total_time+= t
        # elif (x_linha<0 and y_linha>=width)  :
        #     warp_image[0,width-1,0] = 1; #pixels that have atleast one event
        #     warp_image[0,width-1,1] +=1; #sum of events in pixel
        #     warp_image[0,width-1,2] +=t; #sum of the timestamps from events in pixel
        #     cont+=1
        #     total_time+= t
        # elif (x_linha>=0 and x_linha<height and y_linha>=width)  :
        #     warp_image[x_linha,width-1,0] = 1; #pixels that have atleast one event
        #     warp_image[x_linha,width-1,1] +=1; #sum of events in pixel
        #     warp_image[x_linha,width-1,2] +=t; #sum of the timestamps from events in pixel
        #     cont+=1
        #     total_time+= t
        # elif (x_linha>=height and y_linha>=width): #between bounds 
        #     warp_image[height-1,width-1,0] = 1; #pixels that have atleast one event
        #     warp_image[height-1,width-1,1] +=1; #sum of events in pixel
        #     warp_image[height-1,width-1,2] +=t; #sum of the timestamps from events in pixel
        #     cont+=1
        #     total_time+= t
        # elif (x_linha>=height and y_linha>=0 and y_linha<width): #between bounds 
        #     warp_image[height-1,y_linha,0] = 1; #pixels that have atleast one event
        #     warp_image[height-1,y_linha,1] +=1; #sum of events in pixel
        #     warp_image[height-1,y_linha,2] +=t; #sum of the timestamps from events in pixel
        #     cont+=1
        #     total_time+= t
        # elif (x_linha>=height and y_linha<0): #between bounds 
        #     warp_image[height-1,0,0] = 1; #pixels that have atleast one event
        #     warp_image[height-1,0,1] +=1; #sum of events in pixel
        #     warp_image[height-1,0,2] +=t; #sum of the timestamps from events in pixel
        #     cont+=1
        #     total_time+= t
        # elif (x_linha>=0 and x_linha<height and y_linha<0): #between bounds 
        #     warp_image[x_linha,0,0] = 1; #pixels that have atleast one event
        #     warp_image[x_linha,0,1] +=1; #sum of events in pixel
        #     warp_image[x_linha,0,2] +=t; #sum of the timestamps from events in pixel
        #     cont+=1
        #     total_time+= t
    
    mean_t = total_time/cont
    for i in range(0,height):
        for j in range(0,width):
            if (warp_image[i,j,1] != 0):
                warp_image[i,j,2] = warp_image[i,j,2]/warp_image[i,j,1] #save the timestamp average of all the events in that pixel
                warp_image[i,j,3] = warp_image[i,j,2] - mean_t #score [-1,1]
                
            if (warp_image[i,j,3] > 0.002): #threshold 
                warp_image[i,j,4] = 1;
    
    return warp_image

#@profile
def morphology(warp_image):
    
    warp_rodrigues = np.copy(warp_image[:,:,1])                   
    warp_img = np.copy(warp_image[:,:,4])  
    kernel = np.ones((4, 4), np.uint8)
    clean_img = cv2.morphologyEx(warp_img, cv2.MORPH_OPEN, kernel) #morphological operatoins to remove noise in the image

    return warp_img,clean_img,warp_rodrigues

#@profile
def clustering(img,height,width):
    #https://www.section.io/engineering-education/dbscan-clustering-in-python/
    cluster =np.zeros((1,2))
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            if (img[i,j] == 1):
                cluster = np.vstack((cluster,[i,j]))
                  
    if(cluster.shape[0] > 1):
        cluster = np.delete(cluster, 0, 0)  

    dbscan = DBSCAN(eps = 50, min_samples = 2).fit(cluster) # fitting the model
    labels = dbscan.labels_ # getting the labels
    cluster = np.column_stack((cluster, labels))
    cluster[:,2] = cluster[:,2] + 1;
              
    cluster_img = np.zeros([height,width])
    rgb_img = np.zeros([height,width,3])
    cluster_new = cluster.reshape(-1)
    for i in range(0,(len(cluster_new)//3)-1):   
        cluster_img[int(cluster_new[i*3]),int(cluster_new[(i*3)+1])] = cluster_new[(i*3)+2]
        rgb_img[int(cluster_new[i*3]),int(cluster_new[(i*3)+1]),:] = 255
        
    return cluster,cluster_img,rgb_img

#@profile
# def draw_rect(cluster,img):
    
#     cluster = cluster[cluster[:,2].argsort()]
#     list_array = [cluster[cluster[:,2]==k] for k in np.unique(cluster[:,2])] #create a list of arrays for each cluster
#     rect_max_x = []
#     rect_min_x = []
#     rect_max_y = []
#     rect_min_y = []
#     for i in range(0,len(list_array)):
#         rect_max_x = np.append(rect_max_x,max(list_array[i][:,0]))
#         rect_min_x = np.append(rect_min_x,min(list_array[i][:,0]))
#         rect_max_y = np.append(rect_max_y,max(list_array[i][:,1]))
#         rect_min_y = np.append(rect_min_y,min(list_array[i][:,1]))
        
#     cont=0
#     for i in range(0,len(rect_max_x)): 
#         if(rect_max_y[i]-rect_min_y[i] > 5 and rect_max_x[i]-rect_min_x[i] > 5): #Only draws a rectangle for objects with height bigger than 3 pixels
#             cont += 1
#             img = cv2.rectangle(img, (int(rect_min_y[i]),int(rect_min_x[i])), (int(rect_max_y[i]),int(rect_max_x[i])), (0, 255, 0),2) 
#             center_y = ((rect_max_y[i]-rect_min_y[i])//2)+(rect_min_y[i]) #Round number of the center
#             center_x = ((rect_max_x[i]-rect_min_x[i])//2)+(rect_min_x[i])
#             print("Center (height,width) of the Moving Object",cont,": (",center_x,",",center_y,")")
#     return img

#Good idx values to see different perpectives of the data sequences:
    
    #Dynamic: [120,121] 1 dynamic object
    #Dynamic2: [385,386] 1 dynamic object
    #Dynamic3: [148,149] 4 dynamic objects
    #Dynamic Camera: [100,101] 0 dynamic objects
    #Static: [10,11] 0 dynamic objects
    #Dynamic Object: [95,96] 1 dynamic object
    #Dynamic Object2: [259,260] 2 dynamic object
    #Dynamic Object3: [349,350] 1 dynamic object
    #Dynamic Object4: [129,130] 2 dynamic object

data_seq = "Dynamic2"

fname = "C:\\Now\\Data\\" + data_seq + "\\event_data.csv"
fname_imu = "C:\\Now\\Data\\" + data_seq + "\\imu_data.csv"
cloud,idx = read_event_file_txt(fname,0.1,True,False) 
idx = np.append(idx, [cloud.shape[0]]) #appends the last value of idx

width = cloud[-1][0] - cloud[0][0]
print ("")
print ("Input cloud:")
print ("\tWidth: ", width, "seconds and", len(cloud), "events.")
print ("\tIndex size: ", int(len(idx)-2), "points, step = ", width / float(len(idx) + 1), "seconds.")
print ("")

cont_2=0
average=0
for j in range(0,int(len(idx)-2)): 
#for j in range(100,150): 
    bound_1 = idx[j]
    bound_2 = idx[int(j+1)]
    cloud_imu = read_imu_file(fname_imu,cloud[bound_1,0],cloud[bound_2,0])
    sl = cloud[bound_1:bound_2] 

    # width = sl[-1][0] - sl[0][0]
    # print ("Chosen slice:")
    # print ("\tWidth: ", width, "seconds and", len(sl), "events.")
    # print ("\tStarts at: ", cloud[bound_1,0], "seconds and finishes at", cloud[bound_2,0], "seconds.")
    # print ("")
    if (len(cloud_imu) == 0):
        print ("Starts at: ", cloud[bound_1,0], "seconds and ",int(j+1),"points")
        print("\tThis time window does not have IMU information.")
    else:
        height = 720
        width = 1280
        #sl_positive = cut_negative_events(sl)
        #img = show_frame(sl,height,width)
        ego_image = ego_motion(sl,height,width,cloud_imu) 
        warp_img,clean_img,warp_rodrigues = morphology(ego_image) 
        cluster,cluster_img,rgb_img = clustering(clean_img,height,width)
        
        result = timeit.timeit(stmt='ego_motion(sl,height,width,cloud_imu)', globals=globals(), number=1)
        #result = timeit.timeit(stmt='pass',globals=globals(), number=1)
        # print("",result)
        cont_2 = cont_2 + 1 
        average = result + average
        print("",average)
        print("",cont_2)
        # calculate the execution time
        # get the average execution time

print("",average)
print("",cont_2)
average = average/cont_2
print("Average execution time of ",average)

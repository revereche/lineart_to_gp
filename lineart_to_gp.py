from __future__ import division

bl_info = {
    "name": "Line Art to GP",
    "author": "Rev",
    "version": (1, 0),
    "blender": (2, 91, 0),
    "location": "View3D > Sidebar",
    "description": "Imports line art to grease pencil strokes",
    "warning": "",
    "doc_url": "",
    "category": "Import"
}
 
import time, timeit
import bpy, os, sys, re, math, random
from shutil import copyfile
import numpy as np
from scipy import ndimage, spatial
import matplotlib.image as mpimg
from skimage import exposure
from skimage.morphology import medial_axis, dilation, binary_dilation, binary_erosion, erosion, disk
from skimage.util import invert
from skimage.color import rgb2gray, rgba2rgb
from skimage.filters import threshold_otsu, gaussian
from skimage.transform import resize
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from PIL import Image, ImageDraw, ImageFont

import matplotlib.patches as mpatches

from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from skimage.color import rgb2gray
from skimage.draw import line

from scipy.spatial import Voronoi, voronoi_plot_2d

from bpy.types import (Panel, Operator)
from bpy.utils import register_class, unregister_class

default_dir = '/tmp/'

#in case we need to run a function recursively a lot
sys.setrecursionlimit(5000)

def main(context): 
    start_time = timeit.default_timer()
    img_to_strokes()
    print("TOTAL TIME: " + str(timeit.default_timer() - start_time))

def img_to_strokes():
    wm = bpy.context.window_manager

    smoothness = wm.smoothness
    img_dir = wm.img_dir
    img_seq = wm.img_seq
    radius = wm.radius    
    noise = wm.noise    
    thickness = wm.thickness
    strength = wm.strength
    resize = wm.resize
    draw_bounds = wm.draw_bounds
    connect = wm.connect
    overwrite_layer = wm.overwrite_layer
    color_src = wm.color_src
    transparent = wm.transparent
    img_type = wm.img_type
    
    if img_type == "COLOR":
        thickness = wm.col_thickness
        connect = wm.col_connect
        transparent = wm.col_transparent

    obj = bpy.context.view_layer.objects.active
    gp = obj.data
    layers = gp.layers

    files, layer_name, img_dir = dir_test(img_dir, img_seq)
    
    #remove existing layers so we don't get duplicates
    if overwrite_layer:
        for layer in layers:
            try:
                if str(layer.info) == str(layer_name):
                    layers.active = layer
                    bpy.ops.gpencil.layer_remove()   
                if str(layer.info) == str(layer_name) + " boundary":
                    layers.active = layer
                    bpy.ops.gpencil.layer_remove()
            except UnicodeDecodeError:
                continue
        
    layer = gp.layers.new(layer_name)
    layers.active = layer

    for _,img_src in enumerate(files):  
        is_file = img_test(img_src)
        if not is_file:
            continue
        if not is_file:
            continue
        num = get_number(img_src)        
        img_data = Image.open(img_dir + img_src)
        width, height = img_data.size
        #print("image size" + str(img_data.size))
        print("current image: " + str(img_src))
        
        regions = img_to_list(img_data, num, noise, img_type, transparent, smoothness)

        #setting up our canvas
        frame = layer.frames.new(num)
        strokes = frame.strokes
        
        if not resize:
            resize = 0.01

        scale = 4.5 * resize

        width = resize_canvas(width, scale)
        height = resize_canvas(height, scale)

        if connect:
            start_time = timeit.default_timer()
            for i,region in enumerate(regions):
                region = np.array(region)                
                if radius > 4:
                    step = int(radius / 5)
                else:
                    step = 1
                region = region[::step]
                sort_strokes(region, frame, width, height, scale, thickness, strength, radius, color_src)
                
            print("CONNECTED TIME: " + str(timeit.default_timer() - start_time))
          
        else:
            start_time = timeit.default_timer()
            for i,region in enumerate(regions):
                draw_points(region, frame, width, height, scale, thickness, strength, color_src)
            print("DISCONNECTED TIME: " + str(timeit.default_timer() - start_time))

        if draw_bounds:
            draw_boundaries(gp, num, width, height, layer_name)

        if img_type == "SHADING":
            stroke_edit(layer, layers)
            
#DRAW

def traverse(coord_i, coord, coords, iter, radius):
    iter += 1
    stroke = [coord]
    new_coords = coords
    coords.pop(coord_i)
    try: 
        ind, pt = get_neighbor(coord, new_coords, radius)  
    except ValueError:
        return stroke, coords
    except TypeError:
        return stroke, coords
    new_stroke, coords = traverse(ind, pt, coords, iter, radius)
    stroke += new_stroke
    return stroke, coords
  
def get_neighbor(pt, coords, ratio):
    #this is currently also judging distance by pressure...may have to fix that
    distance,index = spatial.KDTree(coords).query(pt)
    if distance <= ratio:
        #print("Neighbor: " + str(coords[index]) + "\n")
        #print("Distance: " + str(distance) + " Index: " + str(index))
        return index, coords[index]
    
def draw_strokes(coords, frame, width, height, scale,thickness, strength, radius, color_src):
    gp = bpy.context.tool_settings.gpencil_paint
    brush = gp.brush
    new_coords = coords.copy()
    try:
        stroke, leftover = traverse(0,new_coords[0],new_coords, 0, radius)
    except IndexError:
        return
    new_stroke = frame.strokes.new()
    new_stroke.display_mode = '3DSPACE' 
    new_stroke.hardness = brush.gpencil_settings.hardness

    strokeLength = len(stroke)
    new_stroke.points.add(count = strokeLength )

    for i,coord in enumerate(stroke):
        offset_width = image_resize(coord[0], scale, width)
        offset_height = -1 * image_resize(coord[1], scale, height)
        if color_src:
            new_stroke.points[i].vertex_color = (coord[3], coord[4], coord[5], 1)
        new_stroke.points[i].co = [offset_width, 0, offset_height]
        new_stroke.points[i].pressure = coord[2] * thickness
        new_stroke.points[i].strength = coord[2] * strength

    slen = 1
    while len(coords) > len(stroke):
        slen += 1
        newer_coords = new_coords.copy()
        try:
            stroke, new_coords = traverse(0,newer_coords[0],newer_coords, 0, radius)
        except IndexError:
            break
        
        new_stroke = frame.strokes.new()
        new_stroke.display_mode = '3DSPACE' 
        new_stroke.hardness = brush.gpencil_settings.hardness
        
        strokeLength = len(stroke)
        new_stroke.points.add(count = strokeLength )

        for i,coord in enumerate(stroke):
            offset_width = image_resize(coord[0], scale, width)
            offset_height = -1 * image_resize(coord[1], scale, height)
            if color_src:
                new_stroke.points[i].vertex_color = (coord[3], coord[4], coord[5], 1)
            new_stroke.points[i].co = [offset_width, 0, offset_height]
            new_stroke.points[i].pressure = coord[2] * thickness
            new_stroke.points[i].strength = coord[2] * strength

def sort_strokes(coords, frame, width, height, scale,thickness, strength, radius, color_src):
    try:       
        ind = np.lexsort((coords[:,1], coords[:,0]))    
    except IndexError:
        return
    sorted_coords = coords[ind]
    sorted_list = sorted_coords.tolist()
    draw_strokes(sorted_list, frame, width, height, scale,thickness, strength, radius, color_src)
    
def draw_points(coords, frame, width, height, scale, thickness, strength, color_src):
    brush = bpy.context.tool_settings.gpencil_paint.brush
    for i,coord in enumerate(coords):
        new_stroke = frame.strokes.new()
        new_stroke.display_mode = '3DSPACE' 
        new_stroke.material_index = bpy.context.object.active_material_index
        new_stroke.hardness = brush.gpencil_settings.hardness

        strokeLength = 1
        new_stroke.points.add(count = strokeLength )
        offset_width = image_resize(coord[0], scale, width)
        offset_height = -1 * image_resize(coord[1], scale, height)
        if color_src:
            new_stroke.points[0].vertex_color = (coord[3], coord[4], coord[5], 1)
        new_stroke.points[0].co = [offset_width, 0, offset_height]
        new_stroke.points[0].pressure = coord[2] * thickness
        new_stroke.points[0].strength = coord[2] * strength


def stroke_edit(edit_layer, layers):
    bpy.ops.object.mode_set(mode='EDIT_GPENCIL')
    
    layers.active = edit_layer
    
    #WARNING: This hasn't been tested on strokes yet
    #we hide all layers but active, but if a layer is already hidden, we don't want to turn it back later
    hidden = []
    for layer in layers:
        if layer.hide:
            hidden.append(layer.info)
            #print(hidden)
        else:
            layer.hide = True
            
    layers.active = edit_layer
    edit_layer.hide = False
    
    bpy.ops.gpencil.hide(unselected=True)      
    bpy.ops.gpencil.select_all(action='SELECT')
    #bpy.ops.gpencil.stroke_subdivide(number_cuts=1, factor=post_smoothness, repeat=1, only_selected=True, smooth_position=True, smooth_thickness=True, smooth_strength=False, smooth_uv=False)

    #smooth first, THEN simplify, or you'll lose edges!! #bpy.ops.gpencil.stroke_smooth(factor=post_smoothness)
    #bpy.ops.gpencil.stroke_simplify(factor=simplify)
    bpy.ops.gpencil.stroke_merge_by_distance(threshold=0.001)

    
    for layer in layers:
        if not layer.info in hidden:
            layer.hide = False
            
    layers.active = edit_layer
    

def draw_boundaries(gp, num, width, height, layer_name):
    
    layer = gp.layers.new(str(layer_name) + " boundary")
    frame = layer.frames.new(int(num))
    strokes = frame.strokes

    new_stroke = frame.strokes.new()
    new_stroke.display_mode = '3DSPACE' 

    # Number of stroke points
    strokeLength = 5
    bound_thickness = 5

    # Add points
    new_stroke.points.add(count = strokeLength )

    new_stroke.points[0].co = [-width,0, -height]
    new_stroke.points[0].pressure = bound_thickness
    new_stroke.points[1].co = [-width,0,height]
    new_stroke.points[1].pressure = bound_thickness
    new_stroke.points[2].co = [width,0,height]
    new_stroke.points[2].pressure = bound_thickness
    new_stroke.points[3].co = [width,0,-height]
    new_stroke.points[3].pressure = bound_thickness
    new_stroke.points[4].co = [-width,0,-height]   
    new_stroke.points[4].pressure = bound_thickness

#VORONOI

#Code to get finite Voronoi edges from http://zderadicka.eu/voronoi-diagrams/
def check_outside(point, bbox):
    point=np.round(point, 4)
    return point[0]<bbox[0] or point[0]>bbox[2] or point[1]< bbox[1] or point[1]>bbox[3]

def calc_shift(point, vector, bbox):
    c=sys.float_info.max
    for l,m in enumerate(bbox):
        a=(float(m)-point[l%2])/vector[l%2]
        if  a>0 and  not check_outside(point+a*vector, bbox):
            if abs(a)<abs(c):
                c=a
    return c if c<sys.float_info.max else 0

def move_point(start, end, bbox):
    vector=end-start
    c=calc_shift(start, vector, bbox)
    if c>0 and c<1:
        start=start+c*vector
        return start

def voronoi3(P, bbox=None): 
    P=np.asarray(P)
    if not bbox:
        xmin=P[:,0].min()
        xmax=P[:,0].max()
        ymin=P[:,1].min()
        ymax=P[:,1].max()
        xrange=(xmax-xmin) * 0.3333333
        yrange=(ymax-ymin) * 0.3333333
        bbox=(xmin-xrange, ymin-yrange, xmax+xrange, ymax+yrange)
    bbox=np.round(bbox,4)
    vor=Voronoi(P)
    center = vor.points.mean(axis=0)
    vs=vor.vertices
    segments=[]
    #print("Vertices: " + str(vor.vertices) + " Ridge Vertices: " + str(vor.ridge_vertices) + " Points: " + str(vor.ridge_points) + " Regions: " + str(vor.regions))
    for i,(istart,iend) in enumerate(vor.ridge_vertices):
        if istart<0 or iend<=0:
            start=vs[istart] if istart>=0 else vs[iend]
            if check_outside(start, bbox) :
                    continue
            first,second = vor.ridge_points[i]
            first,second = vor.points[first], vor.points[second]
            edge= second - first
            vector=np.array([-edge[1], edge[0]])
            midpoint= (second+first)/2
            orientation=np.sign(np.dot(midpoint-center, vector))
            vector=orientation*vector
            c=calc_shift(start, vector, bbox)
            if c is not None:    
                segments.append([start,start+c*vector])
        else:
            start,end=vs[istart], vs[iend]
            if check_outside(start, bbox):
                start=move_point(start,end, bbox)
                if  start is None:
                    continue
            if check_outside(end, bbox):
                end=move_point(end,start, bbox)
                if  end is None:
                    continue
            segments.append( [start, end] )
            
    return segments

def keep_in_bounds(x, dist_on_skel, n):
    #checking to make sure we don't pull coords from outside the image as we go
    x = int(x)
    if x < 0:
        x = 0
    if x > dist_on_skel.shape[n] - 1:
        x = dist_on_skel.shape[n] - 1
    return x

def harris_voronoi(img_data, vor_arr, h_smoothness, dist_on_skel, is_alpha):
    #Using Harris edges and a Voronoi diagram from that to get regions for 
    #every stroke before it hits an angle, so we can just join strokes without
    #getting unwanted black patches from merging points with acute angles
    
    #prep and feed Harris corner data
    h_data = smooth(vor_arr, h_smoothness)
    harris = corner_peaks(corner_harris(h_data), min_distance=1)        
    vor = voronoi3(harris)

    #get connected lines from the finite Voronoi data
    np_img = np.zeros(dist_on_skel.shape)
    for i in vor:        
        x0 = keep_in_bounds(i[0][0], dist_on_skel, 0)
        y0 = keep_in_bounds(i[0][1], dist_on_skel, 1)
        x1 = keep_in_bounds(i[1][0], dist_on_skel, 0)
        y1 = keep_in_bounds(i[1][1], dist_on_skel, 1)
        
        rr, cc = line(x0,y0,x1,y1)
        try:
            np_img[rr, cc] = 1
        except IndexError:
            continue
    
    #prep
    #print("image: " + str(type(np_img[0,0])) + " img_data: " + str(type(img_data[0,0])))
    image = np_img
    image = invert(image)
    img_data = img_data != 0
    
    image = smooth(image, 0.7)
    image *= img_data

    #get regions
    label_image = label(image)
    
    return label_image


#PROCESS

def smooth(img_data, smoothness):
    original = img_data
    if smoothness > 0:
        img_data = gaussian(img_data, smoothness)   
    thresh = threshold_otsu(img_data)
    img_data = img_data > thresh
    
    if smoothness > 0:
        img_data = gaussian(img_data, (smoothness / 4))   
        img_data *= original
        thresh = threshold_otsu(img_data)
        img_data = img_data > thresh
    
    return img_data

def pts_to_list(dist_on_skel, label_image, noise, img_type, color_data):
    region_lst = []
    label_skel = label(dist_on_skel)
    for i,region in enumerate(regionprops(label_image)):
        if region.area > noise:
            #getting our offset
            minr, minc, maxr, maxc = region.bbox
            
            #if you're saving a lot of noise, we'll just increase the area of pixels to grab while we're at it
            #3 is a good value for detail and time tradeoff!!!
            expand = 0
            if noise < 11:
                expand += 1
            if noise == 0:
                expand += 1
            
            minr -= expand
            minc -= expand
            maxr += expand
            maxc += expand
   
            #crop dist_on_skel to region area
            skel_slice = dist_on_skel[minr:maxr, minc:maxc]

            image_lst = []
            for ir, row in enumerate(skel_slice):
                for ic,pressure in enumerate(row):
                    if pressure:
                        x = ic + minc
                        y = ir + minr
                        size = pressure
                        color = color_data[y,x]
                        '''
                        if color_data[y,x,3] < 0.1:
                            continue
                        '''
                        r = color_data[y,x,0] / 255
                        g = color_data[y,x,1] / 255
                        b = color_data[y,x,2] / 255
                        image_lst.append([x, y, size, r,g,b ])

            region_lst.append(image_lst)
    return region_lst

def alpha_test(is_alpha, color_data):
    #if the image is transparent, use that; otherwise, threshold out the white areas
    transparent = False
    a = []
    if is_alpha:
        alpha_mask = color_data[:,:,3]
        a = alpha_mask > 200
        a_ravel = a.ravel()
        
        #if the array is mostly True, then we hardly have any transparency in this image, and it would be better to threshold; so, we look for the frequency of values and sort so the most frequent is first
        unique_a,counts_a = np.unique(a_ravel, return_counts=True)
        sort_a = np.argsort(-counts_a)
        freq_a = a_ravel[sort_a]

        if freq_a[0] == False:
            transparent = True

    #return a, transparent
    return a

def img_to_list(img_data, num, noise, img_type, transparent, smoothness): 

    

    #smoothness = 5
    h_smoothness = smoothness * 1.5
    is_alpha = False
    width, height = img_data.size
    img_data = np.array(img_data)
    img_data = np.array(img_data)
    
    color_data = img_data.copy()
    color_data = exposure.adjust_gamma(color_data, 2)
    
    lst = []
    #strip colors
    try:
        img_data = rgb2gray(rgba2rgb(img_data))
        is_alpha = True
    except ValueError:
        img_data = rgb2gray(img_data)

    '''
    #smooth and threshold Line Art to make sure it isn't basically empty
    smooth_intensity = 0        
    #don't want to smooth the image for the panic test if it's too small
    if width >= 700 and height >= 700:
        smooth_intensity = 2            
        blank_test = smooth(img_data, smooth_intensity)    
        #keep it from panicking if the image is blank
        if is_blank(blank_test):
            print("blank")
            return
    '''
        
    if img_type == "SHADING":

        lst = shading_to_list(img_data, color_data, is_alpha)

    if img_type == "COLOR":
        lst = color_erode(img_data, color_data, is_alpha, transparent, noise, img_type)

    if img_type == "LINEART":
        #make a padded version so the Voronoi output looks better
        vor_arr = np.pad(img_data, pad_width=10, mode='constant', constant_values=1)   

        if is_alpha and transparent:
            img_data = alpha_test(is_alpha, color_data)
        else:
            img_data = smooth(img_data, smoothness)
            img_data = invert(img_data)
        
        #medial axis skeleton, to get point location and thickness
        skel, distance = medial_axis(img_data, return_distance=True)
        dist_on_skel = distance * skel

        #Voronoi diagram from Harris edges, to avoid concave shapes
        label_image = harris_voronoi(img_data, vor_arr, h_smoothness, dist_on_skel, is_alpha)
        
        lst = pts_to_list(dist_on_skel, label_image, noise, img_type, color_data)

    return lst

def shading_to_list(img_data, color_data, is_alpha): 
    label_image = label(img_data)

    a, transparent = alpha_test(is_alpha, color_data)
    if transparent:
        img_data = a
    else:
        smoothness = 9.0
        img_data = smooth(img_data, smoothness)  
        
    img_data = img_data != 0
    
    selem = disk(3)
    img_data = binary_dilation(img_data, selem)
    
    #make lineart out of a solid shape by putting a smaller shape inside a bigger one
    selem = disk(6)
    shrunken = binary_dilation(img_data, selem)
    shrunken = invert(shrunken)
    shrunken = shrunken != 0

    region_lst = []
    for region in regionprops(label_image):
        #gives us square area in original image region is inside
        minr, minc, maxr, maxc = region.bbox
        
        #use the offset to get the position of the region and rip it from the image
        img_slice = img_data[minr:maxr, minc:maxc]
        shrunken_slice = shrunken[minr:maxr, minc:maxc]
        img_slice += shrunken_slice        
        #img_data[minr:maxr, minc:maxc] = img_slice
        
        #medial axis skeleton, to get point location and thickness
        img_slice = invert(img_slice)
        skel, distance = medial_axis(img_slice, return_distance=True)
        skel_slice = distance * skel
        
        image_lst = []
        for ir, row in enumerate(skel_slice):
            for ic,pressure in enumerate(row):
                if pressure:
                    x = ic + minc
                    y = ir + minr
                    size = pressure
                    image_lst.append([x, y, size, 0,0,0])

        region_lst.append(image_lst)
    
    return region_lst


def color_erode(img_data, color_data, is_alpha, transparent, noise, img_type): 
        
    
        
    grayscale = rgb2gray(img_data)
    original_grayscale = grayscale

    grayscale = 10 * np.round_(grayscale, 1)
    grayscale = grayscale.astype(int)

    if is_alpha and transparent:
        alpha_mask = alpha_test(is_alpha, color_data)

    label_image = label(grayscale)
    img_data = grayscale
    
    new_arr = np.zeros(img_data.shape)

    for region in regionprops(label_image):
        if region.area >= 3:
            #gives us square area in original image region is inside
            minr, minc, maxr, maxc = region.bbox
            
            #use the offset to get the position of the region and rip it from the image
            img_slice = region.image
            
            if is_alpha and transparent:
                img_slice *= alpha_mask[minr:maxr, minc:maxc]

            #now erode it and stick it in the new array
            selem = disk(1)
            eroded_slice = binary_erosion(img_slice, selem)
            
            new_arr[minr:maxr, minc:maxc] += eroded_slice
            #new_arr[minr:maxr, minc:maxc] += img_slice


    vor_arr = np.pad(new_arr, pad_width=10, mode='constant', constant_values=1)   

    #medial axis skeleton, to get point location and thickness
    skel, distance = medial_axis(new_arr, return_distance=True)
    dist_on_skel = distance * skel

    #Voronoi diagram from Harris edges, to avoid concave shapes
    h_smoothness = 0
    label_image2 = harris_voronoi(new_arr, vor_arr, h_smoothness, dist_on_skel, is_alpha)
    
    lst = pts_to_list(dist_on_skel, label_image2, noise, img_type, color_data)
        
    return lst

#UTIL
def get_number(f):
    split = f.split('.')
    f = split[0]
    split = re.split(r"(\d+)", f)
    if len(split) > 1:
        num = split[1]
    else:
        num = 0
    try:
        num = int(num)
    except ValueError:
        num = 0
    return num
    
def get_number_nan(f):
    #same function but without safety
    split = f.split('.')
    f = split[0]
    split = re.split(r"(\d+)", f)
    if len(split) > 1:
        num = split[1]
    else:
        num = "nope"
    return num

def get_seq_name(f):
    #get the name for an image sequence
    split = f.split('.')
    f = split[0]
    split = re.split(r"(\d+)", f)
    if len(split) > 1:
        name = split[0]
    else:
        return False
    return name
    
def img_test(img):
    formats = ['png', 'jpg', 'gif']
    img = img.split('.')
    format = img[-1]
    if format.lower() in formats:
        #print("Format:" + str(format.lower()))
        return True
    else:
        print("Wrong file format!")
        return False

def dir_test(img_dir, img_seq):
    if os.path.isdir(img_dir):
        files = os.listdir( img_dir )
        layer_name = img_dir.split('/')
        #for Windows directories
        if len(layer_name) == 1:
            layer_name = img_dir.split('\\')
        #has to be second index back because it apparently returns a blank after the last "/"
        layer_name = layer_name[-2]
        name = layer_name

    else:
        dir_split = img_dir.split('/')
        if len(dir_split) == 1:
            dir_split = img_dir.split('\\')
        img_name = dir_split[-1]   
        dir_name = dir_split[-2]
        name = img_name.split('.')
        name = name[0]
        #print("name: " + str(name))
        img_dir = dir_split[:-1]
        slash = '/'
        img_dir = slash.join(img_dir)
        img_dir += "/"
        #print("img_dir up here: " + str(img_dir))
        
        files = [ img_name ]

        if img_seq:
            is_num = False
            num = get_number_nan(img_name)
            try:
                num = int(num)
                is_num = True
            except ValueError:
                is_num = False
            #if no number, not a sequence; false alarm
            if is_num:
                list_dir = os.listdir(img_dir)
                #print("Img dir now " + str(img_dir))
                files = []
                seq_name = get_seq_name(img_name)
                name = seq_name
                for f in list_dir:
                    if seq_name:
                        if get_seq_name(f) == seq_name:
                            files.append(f)
                    else:
                    #if the file names are all just numbers, we'll get those and add the dir name
                        if get_number(f):
                            name = dir_name
                            files.append(f)
    return files, name, img_dir

def is_blank(arr, threshold=0.2):
    tot = np.float(np.sum(arr))
    if tot/arr.size  > (1-threshold):
       #print("is not blank") 
       return False
    else:
       #print("is blank")
       return True

def resize_canvas(n, scale):
    return (((n * 0.001) / 2) * scale)

def image_resize(n, scale, canvas):
    return ((((n * 0.001 / 2) * scale)) - (canvas / 2)) * 2

def gp_cam(context):

    bpy.ops.object.select_all(action='DESELECT')

    x = np.random.randint(10000)
    gp_data = bpy.data.grease_pencils.new(name=("LineArtGP" + str(x)) )
    gp_ob = bpy.data.objects.new(("LineArtGP" + str(x)), gp_data)
    bpy.context.scene.collection.objects.link(gp_ob)
    cam_data = bpy.data.cameras.new(name=("LineArtGPCam" + str(x)))
    cam_ob = bpy.data.objects.new(("LineArtGPCam" + str(x)), cam_data)
    bpy.context.scene.collection.objects.link(cam_ob)
    
    gp = bpy.data.objects[("LineArtGP" + str(x))]
    cam = bpy.data.objects[("LineArtGPCam" + str(x))]
    gp.name = "LineArtGP"
    cam.name = "LineArtGPCam"
    
    gp.location = (0, 12, 0)
    cam.rotation_euler = (1.57,0,0)
    gp.select_set(True)
    cam.select_set(True)
    bpy.context.view_layer.objects.active = cam
    bpy.ops.object.parent_set(type='OBJECT', keep_transform=False)
    bpy.context.view_layer.objects.active = gp
            
class LineartToGp(bpy.types.Operator):
    """Convert raster lineart to grease pencil strokes"""
    bl_idname = "greasepencil.lineart_to_gp"
    bl_label = "Line Art to GP Operator"
    
    
    @classmethod
    def poll(cls, context):
        if context.active_object.type == "GPENCIL":
            return context.active_object is not None

    def execute(self, context):
        main(context)
        return {'FINISHED'}

class LineartToGpCam(bpy.types.Operator):
    """Initialize camera for ImgToGP to stick a GP object to"""
    bl_idname = "greasepencil.lineart_to_gp_cam"
    bl_label = "Line Art to GP Camera Initialization"
    
    @classmethod
    def poll(cls, context):
        if context.active_object:
            return context.active_object is not None
    
    def execute(self, context):
        gp_cam(context)
        return {'FINISHED'}


class LineartToGpGui(bpy.types.Panel):
    """GUI for Line Art to GP"""
    bl_label = "Line Art to GP"
    bl_idname = "GREASEPENCIL_PT_lineart_to_gp_gui"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    #bl_space_type = 'PROPERTIES'
    #bl_region_type = 'WINDOW'
    bl_category = "LineGP"
    def draw(self, context):
        layout = self.layout
        obj = context.object

        wm = bpy.context.window_manager

        if context.active_object:
            layout.label(text="Follow prealigned camera:")
            column = layout.column()
     
            column.separator()
            split = column.split(align=True)

            column.operator(LineartToGpCam.bl_idname, text="Initialize Camera", icon='OUTLINER_OB_CAMERA')

            if context.active_object.type == "GPENCIL":
            
                layout.label(text="Import images to GP:")            
                column = layout.column()
                
                column.separator()
                split = column.split(align=True)
                
                split.prop(wm, 'img_dir')
                                                    
                column.operator(LineartToGp.bl_idname, text="Convert to GP Strokes", icon='IMAGE_DATA')
      
                column.separator()
                split = column.split(align=True)
                split.prop(wm, 'img_seq')
                split.prop(wm, 'draw_bounds')
                column.separator()
                split = column.split(align=True)
                split.prop(wm, 'overwrite_layer')
                
                if wm.img_type == "COLOR":
                    split.prop(wm, 'col_connect')
                else:
                    split.prop(wm, 'connect')
                
                column.separator()
                split = column.split(align=True)
                split.prop(wm, 'color_src')
                
                if wm.img_type == "COLOR":
                    split.prop(wm, 'col_transparent')
                else:
                    split.prop(wm, 'transparent')

                layout.label(text="Image Preprocessing:")
                column = layout.column()
                
                column.separator()
                split = column.split(align=True)
                split.prop(wm, 'smoothness')
                
                column.separator()
                split = column.split(align=True)
                split.prop(wm, 'noise')
                
                column.separator()
                split = column.split(align=True)
                split.prop(wm, 'radius')

                column.separator()
                split = column.split(align=True)
                split.prop(wm, 'resize')

                row = layout.row()
                scene = context.scene

                layout.label(text="Image Type:")
                column = layout.column()
                
                column.separator()
                split = column.split(align=True)
                split.prop(wm, 'img_type')
                
                layout.label(text="Stroke Properties:")
                column = layout.column()
                
                column.separator()
                split = column.split(align=True)
                
                if wm.img_type == "COLOR":
                    split.prop(wm, 'col_thickness')
                else:
                    split.prop(wm, 'thickness')
                
                split.prop(wm, 'strength')

bpy.types.WindowManager.img_dir=bpy.props.StringProperty(name='', subtype='FILE_PATH',
        default=default_dir, description='Select file or folder')

bpy.types.WindowManager.img_seq=bpy.props.BoolProperty(name='Image Sequence',
        default=False, description='Open numbered image sequence in folder')
bpy.types.WindowManager.draw_bounds=bpy.props.BoolProperty(name='Draw Bounds',
        default=False, description='Show image boundaries')
bpy.types.WindowManager.overwrite_layer=bpy.props.BoolProperty(name='Overwrite Layer',
        default=True, description='Overwrite existing layer on new import')
        
bpy.types.WindowManager.color_src=bpy.props.BoolProperty(name='Color From Image',
        default=True, description='Otherwise use active material')
bpy.types.WindowManager.transparent=bpy.props.BoolProperty(name='Transparent',
        default=False, description='Otherwise thresholds white')
        
#custom default var for color mode
bpy.types.WindowManager.col_transparent=bpy.props.BoolProperty(name='Transparent',
        default=True, description='Otherwise thresholds white')

#stroke options
bpy.types.WindowManager.thickness=bpy.props.FloatProperty(name='Thickness',
        min=.01, max=100, default=9, description='Grease Pencil stroke thickness')
#different var for color mode
bpy.types.WindowManager.col_thickness=bpy.props.FloatProperty(name='Thickness',
min=.01, max=100, default=9.8, description='Grease Pencil stroke thickness')

bpy.types.WindowManager.strength=bpy.props.FloatProperty(name='Strength',
        min=.01, max=100, default=1, description='Grease Pencil stroke strength')
        
bpy.types.WindowManager.connect=bpy.props.BoolProperty(name='Connect Points', default=True, description='Connect points as strokes')
#var for color mode
bpy.types.WindowManager.col_connect=bpy.props.BoolProperty(name='Connect Points', default=False, description='Connect points as strokes')

#image preprocessing
bpy.types.WindowManager.smoothness=bpy.props.FloatProperty(name='Smoothing',
        min=0, max=20, default=0, description='Smooth sketchy images')  
bpy.types.WindowManager.noise=bpy.props.FloatProperty(name='Noise Filter',
        min=0, max=200, default=3, description='Ignore small pieces')  
bpy.types.WindowManager.radius=bpy.props.FloatProperty(name='Skip Points',
        min=0, max=40, default=5, description='Skip points for faster processing, lower quality')  

bpy.types.WindowManager.resize=bpy.props.FloatProperty(name='Resize (Percentage)',
        min=0, max=5, default=1, description='Resize image relative to canvas')

bpy.types.WindowManager.img_type=bpy.props.EnumProperty(name='', items=[('LINEART','Line Art', 
        'Black and white line art'),('COLOR', 'Color','Flat colors to go under line art')], 
        description="Type of image we're importing", default='LINEART')

'''
#image type
bpy.types.WindowManager.img_type=bpy.props.EnumProperty(name='', items=[('LINEART','Line Art', 
        'Black and white line art'),('COLOR', 'Color','Flat colors to go under line art'), ('SHADING', 'Shading/Lighting','Shading or lighting, all one color')], 
        description="Type of image we're importing", default='LINEART')
'''

_classes = [
    LineartToGp,
    LineartToGpGui,
    LineartToGpCam
]

def register():
    for cls in _classes:
        register_class(cls)

def unregister():
    for cls in _classes:
        unregister_class(cls)

if __name__ == "__main__":
    register()

    # test call
    #bpy.ops.greasepencil.lineart_to_gp()

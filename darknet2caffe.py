import sys
sys.path.append('/data/xiaohang/caffe/python')
import caffe
import numpy as np
from collections import OrderedDict
from cfg import *
from prototxt import *

def darknet2caffe(cfgfile, weightfile, protofile, caffemodel):
    net_info = cfg2prototxt(cfgfile)
    fake_protofile = protofile +'_fake'
    save_prototxt(net_info , fake_protofile, region=False,fake=True)
    print('save prototxt to %s' % protofile)
    save_prototxt(net_info , protofile, region=True,fake = False)
    net = caffe.Net(fake_protofile, caffe.TEST)
    params = net.params

    blocks = parse_cfg(cfgfile)
#######
    fp = open(weightfile, 'rb')
    header = np.fromfile(fp, count=4, dtype=np.int32)
       #add by qfdong to adapt to different net.seen
    if header[0]*10 + header[1] >= 2:
        header = np.fromfile(fp, count=1, dtype=np.int32)

    buf = np.fromfile(fp, dtype = np.float32)
    fp.close()
######

    layers = []
    layer_id = 1
    start = 0
    for block in blocks:
        if start >= buf.size:
            break

        if block['type'] == 'net':
            continue
        elif block['type'] == 'convolutional':
            batch_normalize = int(block['batch_normalize'])
            if block.has_key('name'):
                conv_layer_name = block['name']
                bn_layer_name = '%s-bn' % block['name']
                scale_layer_name = '%s-scale' % block['name']
            else:
                conv_layer_name = 'layer%d-conv' % layer_id
                bn_layer_name = 'layer%d-bn' % layer_id
                scale_layer_name = 'layer%d-scale' % layer_id

            if batch_normalize:
                start = load_conv_bn2caffe(buf, start, params[conv_layer_name], params[bn_layer_name], params[scale_layer_name])
            else:
                start = load_conv2caffe(buf, start, params[conv_layer_name])
            layer_id = layer_id+1
        elif block['type'] == 'connected':
            if block.has_key('name'):
                fc_layer_name = block['name']
            else:
                fc_layer_name = 'layer%d-fc' % layer_id
            start = load_fc2caffe(buf, start, params[fc_layer_name])
            layer_id = layer_id+1
        elif block['type'] == 'maxpool':
            layer_id = layer_id+1
        elif block['type'] == 'avgpool':
            layer_id = layer_id+1
        elif block['type'] == 'region':
            layer_id = layer_id + 1
        elif block['type'] == 'route':
            layer_id = layer_id + 1
        elif block['type'] == 'shortcut':
            layer_id = layer_id + 1
        elif block['type'] == 'softmax':
            layer_id = layer_id + 1
        elif block['type'] == 'cost':
            layer_id = layer_id + 1
        elif block['type'] == 'reorg':
            layer_id = layer_id + 1
        else:
            print('unknow layer type %s ' % block['type'])
            layer_id = layer_id + 1

    print('save caffemodel to %s' % caffemodel)
    net.save(caffemodel)

def load_conv2caffe(buf, start, conv_param):
    weight = conv_param[0].data
    bias = conv_param[1].data
    conv_param[1].data[...] = np.reshape(buf[start:start+bias.size], bias.shape);   start = start + bias.size
    conv_param[0].data[...] = np.reshape(buf[start:start+weight.size], weight.shape); start = start + weight.size
    return start

def load_fc2caffe(buf, start, fc_param):
    weight = fc_param[0].data
    bias = fc_param[1].data
    fc_param[1].data[...] = np.reshape(buf[start:start+bias.size], bias.shape);   start = start + bias.size
    fc_param[0].data[...] = np.reshape(buf[start:start+weight.size], weight.shape); start = start + weight.size
    return start


def load_conv_bn2caffe(buf, start, conv_param, bn_param, scale_param):
    conv_weight = conv_param[0].data
    running_mean = bn_param[0].data
    running_var = bn_param[1].data
    scale_weight = scale_param[0].data
    scale_bias = scale_param[1].data

    scale_param[1].data[...] = np.reshape(buf[start:start+scale_bias.size], scale_bias.shape); start = start + scale_bias.size
    scale_param[0].data[...] = np.reshape(buf[start:start+scale_weight.size], scale_weight.shape); start = start + scale_weight.size
    bn_param[0].data[...] = np.reshape(buf[start:start+running_mean.size], running_mean.shape); start = start + running_mean.size
    bn_param[1].data[...] = np.reshape(buf[start:start+running_var.size], running_var.shape); start = start + running_var.size
    bn_param[2].data[...] = np.array([1.0])
    conv_param[0].data[...] = np.reshape(buf[start:start+conv_weight.size], conv_weight.shape); start = start + conv_weight.size
    return start

def cfg2prototxt(cfgfile):
    blocks = parse_cfg(cfgfile)

    layers = []
    layer_dims = []
    props = OrderedDict() 
    bottom = 'data'
    layer_id = 1
    topnames = dict()
    for block in blocks:
        if block['type'] == 'net':
            props['name'] = 'darknet2caffe'
            props['input'] = 'data'
            props['input_dim'] = ['1']
            props['input_dim'].append(block['channels'])
            props['input_dim'].append(block['height'])
            props['input_dim'].append(block['width'])

            blob_dims = {}
            blob_dims['channels']=int(block['channels'])
            blob_dims['height'] = int(block['height'])
            blob_dims['width'] = int(block['width'])
            layer_dims.append(blob_dims)

            continue
        elif block['type'] == 'convolutional':
            conv_layer = OrderedDict()
            conv_layer['bottom'] = bottom
            if block.has_key('name'):
                conv_layer['top'] = block['name']
                conv_layer['name'] = block['name']
            else:
                conv_layer['top'] = 'layer%d-conv' % layer_id
                conv_layer['name'] = 'layer%d-conv' % layer_id
            conv_layer['type'] = 'Convolution'
            convolution_param = OrderedDict()
            convolution_param['num_output'] = block['filters']
            convolution_param['kernel_size'] = block['size']
            if block['pad'] == '1':
                convolution_param['pad'] = str(int(convolution_param['kernel_size'])/2)
            convolution_param['stride'] = block['stride']
            if block['batch_normalize'] == '1':
                convolution_param['bias_term'] = 'false'
            else:
                convolution_param['bias_term'] = 'true'
            conv_layer['convolution_param'] = convolution_param
            layers.append(conv_layer)
            bottom = conv_layer['top']

            blob_dims = {}
            pad = int(convolution_param['pad'])
            kernel_size = int(convolution_param['kernel_size'])
            stride = int(convolution_param['stride'])
            blob_dims['width'] =(layer_dims[layer_id - 1]['width']+pad*2- kernel_size )/stride+1
            blob_dims['height'] = (layer_dims[layer_id - 1]['height']+pad*2- kernel_size )/stride+1
            blob_dims['channels'] = int(convolution_param['num_output'])
            layer_dims.append(blob_dims)



            if block['batch_normalize'] == '1':
                bn_layer = OrderedDict()
                bn_layer['bottom'] = bottom
                bn_layer['top'] = bottom
                if block.has_key('name'):
                    bn_layer['name'] = '%s-bn' % block['name']
                else:
                    bn_layer['name'] = 'layer%d-bn' % layer_id
                bn_layer['type'] = 'BatchNorm'
                batch_norm_param = OrderedDict()
                batch_norm_param['use_global_stats'] = 'true'
                bn_layer['batch_norm_param'] = batch_norm_param
                layers.append(bn_layer)

                scale_layer = OrderedDict()
                scale_layer['bottom'] = bottom
                scale_layer['top'] = bottom
                if block.has_key('name'):
                    scale_layer['name'] = '%s-scale' % block['name']
                else:
                    scale_layer['name'] = 'layer%d-scale' % layer_id
                scale_layer['type'] = 'Scale'
                scale_param = OrderedDict()
                scale_param['bias_term'] = 'true'
                scale_layer['scale_param'] = scale_param
                layers.append(scale_layer)

            if block['activation'] != 'linear':
                relu_layer = OrderedDict()
                relu_layer['bottom'] = bottom
                relu_layer['top'] = bottom
                if block.has_key('name'):
                    relu_layer['name'] = '%s-act' % block['name']
                else:
                    relu_layer['name'] = 'layer%d-act' % layer_id
                relu_layer['type'] = 'ReLU'
                if block['activation'] == 'leaky':
                    relu_param = OrderedDict()
                    relu_param['negative_slope'] = '0.1'
                    relu_layer['relu_param'] = relu_param
                layers.append(relu_layer)
            topnames[layer_id] = bottom
            layer_id = layer_id+1
        elif block['type'] == 'maxpool':
            max_layer = OrderedDict()
            max_layer['bottom'] = bottom
            if block.has_key('name'):
                max_layer['top'] = block['name']
                max_layer['name'] = block['name']
            else:
                max_layer['top'] = 'layer%d-maxpool' % layer_id
                max_layer['name'] = 'layer%d-maxpool' % layer_id
            max_layer['type'] = 'Pooling'
            pooling_param = OrderedDict()
            pooling_param['kernel_size'] = block['size']
            pooling_param['stride'] = block['stride']
            pooling_param['pool'] = 'MAX'
            if block.has_key('pad') and int(block['pad']) == 1:
                pooling_param['pad'] = str((int(block['size'])-1)/2)
            max_layer['pooling_param'] = pooling_param
            layers.append(max_layer)
            bottom = max_layer['top']
            topnames[layer_id] = bottom

            #######
            blob_dims = {}
            pad = 0
            if pooling_param.has_key('pad'):
                pad = int(pooling_param['pad'] )
            kernel_size = int(pooling_param['kernel_size'])
            stride=int(pooling_param['stride'])
            blob_dims['width'] = (layer_dims[layer_id - 1]['width'] + pad * 2 - kernel_size) / stride + 1
            blob_dims['height'] = (layer_dims[layer_id - 1]['height'] + pad * 2 - kernel_size) / stride + 1
            blob_dims['channels'] = layer_dims[layer_id - 1]['channels']
            layer_dims.append(blob_dims)

            layer_id = layer_id + 1
        elif block['type'] == 'avgpool':
            avg_layer = OrderedDict()
            avg_layer['bottom'] = bottom
            if block.has_key('name'):
                avg_layer['top'] = block['name']
                avg_layer['name'] = block['name']
            else:
                avg_layer['top'] = 'layer%d-avgpool' % layer_id
                avg_layer['name'] = 'layer%d-avgpool' % layer_id
            avg_layer['type'] = 'Pooling'
            pooling_param = OrderedDict()
            pooling_param['kernel_size'] = 7
            pooling_param['stride'] = 1
            pooling_param['pool'] = 'AVE'
            avg_layer['pooling_param'] = pooling_param
            layers.append(avg_layer)
            bottom = avg_layer['top']
            topnames[layer_id] = bottom

            ###########
            blob_dims = {}
            pad = int(pooling_param['pad'])
            kernel_size = int(pooling_param['kernel_size'])
            stride = int(pooling_param['stride'])
            blob_dims['width'] = (layer_dims[layer_id - 1]['width'] + pad * 2 -kernel_size) / stride + 1
            blob_dims['height'] = (layer_dims[layer_id - 1]['height']+ pad * 2 -kernel_size) / stride + 1
            blob_dims['channels'] = layer_dims[layer_id - 1]['channel']
            layer_dims.append(blob_dims)
            layer_id = layer_id + 1
        elif block['type'] == 'region':
            if True:
                region_layer = OrderedDict()
                region_layer['bottom'] = bottom
                if block.has_key('name'):
                    region_layer['top'] = block['name']
                    region_layer['name'] = block['name']
                else:
                    region_layer['top'] = 'layer%d-region' % layer_id
                    region_layer['name'] = 'layer%d-region' % layer_id
                region_layer['type'] = 'DetectionOutput'
                region_param = OrderedDict()
                biases = [float(i) for i in block['anchors'].split(',')]
                region_param['side'] = 13
                region_param['num_classes'] = block['classes']
                region_param['coords'] = block['coords']
                region_param['num_box'] = block['num']
                region_param['confidence_threshold'] = 0.24
                region_param['nms_threshold'] = 0.4
                region_param['biases'] = biases
                region_layer['detection_output_param'] = region_param
                layers.append(region_layer)
                bottom = region_layer['top']
            topnames[layer_id] = bottom
            layer_id = layer_id + 1
        elif block['type'] == 'route':
            concat_index = [int(i) for i in block['layers'].split(',')]
            concat_layer = OrderedDict()
            bottom_tmp=[]

            ##
            width =0
            height=0
            channel=0

            for index_num in range(len(concat_index)):
                prev_layer_id = layer_id + concat_index[index_num]
                bottom_tmp.append(topnames[prev_layer_id])
                width =layer_dims[prev_layer_id]['width']
                height = layer_dims[prev_layer_id]['height']
                channel +=layer_dims[prev_layer_id]['channels']
            concat_layer['bottom']=bottom_tmp
            if block.has_key('name'):
                concat_layer['top'] = block['name']
                concat_layer['name'] = block['name']
            else:
                concat_layer['top'] = 'layer%d-concat' % layer_id
                concat_layer['name'] = 'layer%d-concat' % layer_id
            concat_layer['type'] = 'Concat'

            bottom = concat_layer['top']
            topnames[layer_id] = bottom_tmp


            layers.append(concat_layer)
            ###########
            blob_dims = {}
            blob_dims['width'] = width
            blob_dims['height'] = height
            blob_dims['channels'] =channel;
            layer_dims.append(blob_dims)

            layer_id = layer_id + 1


        elif block['type'] == 'reorg':
            reorg_layer = OrderedDict()
            reorg_layer['bottom'] = bottom
            if block.has_key('name'):
                reorg_layer['top'] = block['name']
                reorg_layer['name'] = block['name']
            else:
                reorg_layer['top'] = 'layer%d-reorg' % layer_id
                reorg_layer['name'] = 'layer%d-reorg' % layer_id
            reorg_layer['type'] = 'Reorg'
            reorg_param = OrderedDict()
            reorg_param['stride'] = block['stride']
            reorg_layer['reorg_param'] = reorg_param
            bottom = reorg_layer['top']
            topnames[layer_id] = bottom
            layers.append(reorg_layer)
            ###########
            blob_dims ={}
            stride =int(reorg_param['stride'])
            blob_dims['width'] = layer_dims[layer_id - 1]['width'] /stride
            blob_dims['height'] = layer_dims[layer_id - 1]['height'] /stride
            blob_dims['channels'] = layer_dims[layer_id - 1]['channels']*stride*stride;
            layer_dims.append(blob_dims)
            print(blob_dims)
            layer_id = layer_id + 1

        elif block['type'] == 'shortcut':
            prev_layer_id1 = layer_id + int(block['from'])
            prev_layer_id2 = layer_id - 1
            bottom1 = topnames[prev_layer_id1]
            bottom2= topnames[prev_layer_id2]
            shortcut_layer = OrderedDict()
            shortcut_layer['bottom'] = [bottom1, bottom2]
            if block.has_key('name'):
                shortcut_layer['top'] = block['name']
                shortcut_layer['name'] = block['name']
            else:
                shortcut_layer['top'] = 'layer%d-shortcut' % layer_id
                shortcut_layer['name'] = 'layer%d-shortcut' % layer_id
            shortcut_layer['type'] = 'Eltwise'
            eltwise_param = OrderedDict()
            eltwise_param['operation'] = 'SUM'
            shortcut_layer['eltwise_param'] = eltwise_param
            layers.append(shortcut_layer)
            bottom = shortcut_layer['top']
 
            if block['activation'] != 'linear':
                relu_layer = OrderedDict()
                relu_layer['bottom'] = bottom
                relu_layer['top'] = bottom
                if block.has_key('name'):
                    relu_layer['name'] = '%s-act' % block['name']
                else:
                    relu_layer['name'] = 'layer%d-act' % layer_id
                relu_layer['type'] = 'ReLU'
                if block['activation'] == 'leaky':
                    relu_param = OrderedDict()
                    relu_param['negative_slope'] = '0.1'
                    relu_layer['relu_param'] = relu_param
                layers.append(relu_layer)
            topnames[layer_id] = bottom
            layer_id = layer_id+1           
            
        elif block['type'] == 'connected':
            fc_layer = OrderedDict()
            fc_layer['bottom'] = bottom
            if block.has_key('name'):
                fc_layer['top'] = block['name']
                fc_layer['name'] = block['name']
            else:
                fc_layer['top'] = 'layer%d-fc' % layer_id
                fc_layer['name'] = 'layer%d-fc' % layer_id
            fc_layer['type'] = 'InnerProduct'
            fc_param = OrderedDict()
            fc_param['num_output'] = int(block['output'])
            fc_layer['inner_product_param'] = fc_param
            layers.append(fc_layer)
            bottom = fc_layer['top']

            if block['activation'] != 'linear':
                relu_layer = OrderedDict()
                relu_layer['bottom'] = bottom
                relu_layer['top'] = bottom
                if block.has_key('name'):
                    relu_layer['name'] = '%s-act' % block['name']
                else:
                    relu_layer['name'] = 'layer%d-act' % layer_id
                relu_layer['type'] = 'ReLU'
                if block['activation'] == 'leaky':
                    relu_param = OrderedDict()
                    relu_param['negative_slope'] = '0.1'
                    relu_layer['relu_param'] = relu_param
                layers.append(relu_layer)
            topnames[layer_id] = bottom
            layer_id = layer_id+1
        else:
            print('unknow layer type %s ' % block['type'])
            topnames[layer_id] = bottom
            layer_id = layer_id + 1

    net_info = OrderedDict()
    net_info['props'] = props
    net_info['layers'] = layers
    net_info['layer_dims'] = layer_dims
    return net_info

if __name__ == '__main__':
    import sys
    if not(len(sys.argv) == 5 or len(sys.argv) == 3):
        print('try:')
        print('python darknet2caffe.py tiny-yolo-voc.cfg tiny-yolo-voc.weights tiny-yolo-voc.prototxt tiny-yolo-voc.caffemodel')
        print('')
        print('please add name field for each block to avoid generated name')
        exit()
    
    #add by qfdong,change to cpu mode
    caffe.set_mode_cpu()
    if len(sys.argv) == 5:
        cfgfile = sys.argv[1]
    #net_info = cfg2prototxt(cfgfile)
    #print_prototxt(net_info)
    #save_prototxt(net_info, 'tmp.prototxt')
        weightfile = sys.argv[2]
        protofile = sys.argv[3]
        caffemodel = sys.argv[4]
    if len(sys.argv) == 3:
        cfgfile = sys.argv[1]
        protofile = sys.argv[2]
        weightfile=0
        caffemodel=0

    darknet2caffe(cfgfile, weightfile, protofile, caffemodel)

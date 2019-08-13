import tensorflow as tf
import numpy as np
import tf_util


def input_transform_net(edge_feature, is_training, bn_decay=None, K=3, is_dist=False):
    batch_size = edge_feature.get_shape()[0].value
    num_point = edge_feature.get_shape()[1].value

    # input_image = tf.expand_dims(point_cloud, -1)
    net = tf_util.conv2d(edge_feature, 64, [1,1],
                padding='VALID', stride=[1,1],
                bn=True, is_training=is_training,
                scope='tconv1', bn_decay=bn_decay, is_dist=is_dist)
    net = tf_util.conv2d(net, 128, [1,1],
                padding='VALID', stride=[1,1],
                bn=True, is_training=is_training,
                scope='tconv2', bn_decay=bn_decay, is_dist=is_dist)

    net = tf.reduce_max(net, axis=-2, keep_dims=True)

    net = tf_util.conv2d(net, 1024, [1,1],
                padding='VALID', stride=[1,1],
                bn=True, is_training=is_training,
                scope='tconv3', bn_decay=bn_decay, is_dist=is_dist)
    net = tf_util.max_pool2d(net, [num_point,1],
                padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                    scope='tfc1', bn_decay=bn_decay,is_dist=is_dist)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                    scope='tfc2', bn_decay=bn_decay,is_dist=is_dist)

    with tf.variable_scope('transform_XYZ') as sc:
        # assert(K==3)
        with tf.device('/cpu:0'):
            weights = tf.get_variable('weights', [256, K*K],
                        initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
            biases = tf.get_variable('biases', [K*K],
                        initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
        biases += tf.constant(np.eye(K).flatten(), dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, K, K])
    return transform


def get_model(input_tensor, is_training, bn_decay = None):    
    weight_decay = 0.0
    num_point = input_tensor.get_shape()[1].value
    
    k = 10

    # #Transform Net
    # adj_matrix = tf_util.pairwise_distance(input_tensor)
    # nn_idx = tf_util.knn(adj_matrix, k=k)
    # edge_feature = tf_util.get_edge_feature(input_tensor, nn_idx=nn_idx, k=k)

    # with tf.variable_scope('transform_net1'):
    #     transform = input_transform_net(edge_feature, is_training, bn_decay, K=input_tensor.get_shape()[2])

    # input_tensor_transformed = tf.matmul(input_tensor, transform)


    #Transform Net
    adj_matrix = tf_util.pairwise_distance(input_tensor)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(input_tensor, nn_idx=nn_idx, k=k)

    out1_1 = tf_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='one/adj_conv1', bn_decay=bn_decay, is_dist=True)
    
    out1_2 = tf_util.conv2d(out1_1, 64, [1,1],
                        padding='VALID', stride=[1,1],
                        bn=True, is_training=is_training, weight_decay=weight_decay,
                        scope='one/adj_conv2', bn_decay=bn_decay, is_dist=True)

        
    out1_3 = tf_util.conv2d(out1_2, 64, [1,1],
                        padding='VALID', stride=[1,1],
                        bn=True, is_training=is_training, weight_decay=weight_decay,
                        scope='one/adj_conv3', bn_decay=bn_decay, is_dist=True)

    net_1 = tf.reduce_max(out1_3, axis=-2, keepdims=True)



    adj = tf_util.pairwise_distance(net_1)
    nn_idx = tf_util.knn(adj, k=k)
    edge_feature = tf_util.get_edge_feature(net_1, nn_idx=nn_idx, k=k)

    out2_1 = tf_util.conv2d(edge_feature, 64, [1,1],
                        padding='VALID', stride=[1,1],
                        bn=True, is_training=is_training, weight_decay=weight_decay,
                        scope='two/adj_conv1', bn_decay=bn_decay, is_dist=True)

    out2_2 = tf_util.conv2d(out2_1, 64, [1,1],
                        padding='VALID', stride=[1,1],
                        bn=True, is_training=is_training, weight_decay=weight_decay,
                        scope='two/adj_conv2', bn_decay=bn_decay, is_dist=True)

    out2_3 = tf_util.conv2d(out2_2, 64, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training, weight_decay=weight_decay,
                            scope='two/adj_conv3', bn_decay=bn_decay, is_dist=True)
                            
    net_2 = tf.reduce_max(out2_3, axis=-2, keepdims=True)

      

    adj = tf_util.pairwise_distance(net_2)
    nn_idx = tf_util.knn(adj, k=k)
    edge_feature = tf_util.get_edge_feature(net_2, nn_idx=nn_idx, k=k)

    out3_1 = tf_util.conv2d(edge_feature, 64, [1,1],
                        padding='VALID', stride=[1,1],
                        bn=True, is_training=is_training, weight_decay=weight_decay,
                        scope='three/adj_conv1', bn_decay=bn_decay, is_dist=True)


    out3_2 = tf_util.conv2d(out3_1, 64, [1,1],
                        padding='VALID', stride=[1,1],
                        bn=True, is_training=is_training, weight_decay=weight_decay,
                        scope='three/adj_conv2', bn_decay=bn_decay, is_dist=True)


    net_3 = tf.reduce_max(out3_2, axis=-2, keepdims=True)



    out7 = tf_util.conv2d(tf.concat([net_1, net_2, net_3], axis=-1), 1024, [1, 1], 
                        padding='VALID', stride=[1,1],
                        bn=True, is_training=is_training,
                        scope='adj_conv7', bn_decay=bn_decay, is_dist=True)

    out_max = tf_util.max_pool2d(out7, [num_point, 1], padding='VALID', scope='maxpool')


    expand = tf.tile(out_max, [1, num_point, 1, 1])

    concat = tf.concat(axis=3, values=[expand, 
                                        net_1,
                                        net_2,
                                        net_3])

    # CONV 
    net = tf_util.conv2d(concat, 512, [1,1], padding='VALID', stride=[1,1],
                bn=True, is_training=is_training, scope='seg/conv1', is_dist=True)
    # net = tf_util.conv2d(net, 256, [1,1], padding='VALID', stride=[1,1],
    #             bn=True, is_training=is_training, scope='seg/conv2', is_dist=True)
    # net = tf_util.conv2d(net, 128, [1,1], padding='VALID', stride=[1,1],
    #             bn=True, is_training=is_training, scope='seg/conv3', is_dist=True)
    # net = tf_util.conv2d(net, 64, [1,1], padding='VALID', stride=[1,1],
    #             bn=True, is_training=is_training, scope='seg/conv4', is_dist=True)
    # net = tf_util.conv2d(net, 32, [1,1], padding='VALID', stride=[1,1],
    #             bn=True, is_training=is_training, scope='seg/conv5', is_dist=True)    
    
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    
    net = tf_util.conv2d(net, 16, [1,1], padding='VALID', stride=[1,1],
                activation_fn=None, scope='seg/output', is_dist=True)


    net = tf.squeeze(net, [2])


    return net


    

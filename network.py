import tensorflow as tf
import tf_util

def get_model(input_tensor, is_training, bn_decay = None):    
    weight_decay = 0.0
    num_point = input_tensor.get_shape()[1].value
    
    k = 5

    adj_matrix = tf_util.pairwise_distance(input_tensor)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    edge_feature = tf_util.get_edge_feature(input_tensor, nn_idx=nn_idx, k=k)

    out1 = tf_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='adj_conv1', bn_decay=bn_decay, is_dist=True)
    
    out2 = tf_util.conv2d(out1, 64, [1,1],
                        padding='VALID', stride=[1,1],
                        bn=True, is_training=is_training, weight_decay=weight_decay,
                        scope='adj_conv2', bn_decay=bn_decay, is_dist=True)

    net_1 = tf.reduce_max(out2, axis=-2, keepdims=True)



    adj = tf_util.pairwise_distance(net_1)
    nn_idx = tf_util.knn(adj, k=k)
    edge_feature = tf_util.get_edge_feature(net_1, nn_idx=nn_idx, k=k)

    out3 = tf_util.conv2d(edge_feature, 64, [1,1],
                        padding='VALID', stride=[1,1],
                        bn=True, is_training=is_training, weight_decay=weight_decay,
                        scope='adj_conv3', bn_decay=bn_decay, is_dist=True)

    out4 = tf_util.conv2d(out3, 64, [1,1],
                        padding='VALID', stride=[1,1],
                        bn=True, is_training=is_training, weight_decay=weight_decay,
                        scope='adj_conv4', bn_decay=bn_decay, is_dist=True)
    
    net_2 = tf.reduce_max(out4, axis=-2, keepdims=True)

      

    adj = tf_util.pairwise_distance(net_2)
    nn_idx = tf_util.knn(adj, k=k)
    edge_feature = tf_util.get_edge_feature(net_2, nn_idx=nn_idx, k=k)

    out5 = tf_util.conv2d(edge_feature, 64, [1,1],
                        padding='VALID', stride=[1,1],
                        bn=True, is_training=is_training, weight_decay=weight_decay,
                        scope='adj_conv5', bn_decay=bn_decay, is_dist=True)


    net_3 = tf.reduce_max(out5, axis=-2, keepdims=True)



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
    net = tf_util.conv2d(net, 256, [1,1], padding='VALID', stride=[1,1],
                bn=True, is_training=is_training, scope='seg/conv2', is_dist=True)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp1')
    
    net = tf_util.conv2d(net, 2, [1,1], padding='VALID', stride=[1,1],
                activation_fn=None, scope='seg/conv3', is_dist=True)


    net = tf.squeeze(net, [2])

    return net


    

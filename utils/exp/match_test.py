
def att_match(att_roi_tuple, pre_att_roi_tuple, pooling_size=30):
    match_list = [None] * len(att_roi_tuple)
    if not pre_att_roi_tuple:
        return match_list
    else:
        xycls_dis = np.zeros(len(att_roi_tuple), len(pre_att_roi_tuple))
        for num, obj in enumerate(att_roi_tuple):
            obj[0] = [F.upsample(roi, (pooling_size,pooling_size), mode='bilinear') for roi in obj[0]]
            obj_x_min, obj_y_min, obj_x_max, obj_y_max, obj_cls = obj[1:]
            for pre_num, pre_obj in enumerate(pre_att_roi_tuple):
                if pre_num == 0:
                    pre_obj[0] = [F.upsample(preroi, (pooling_size,pooling_size)) for preroi in pre_att_roi]
                preobj_x_min, preobj_y_min, preobj_x_max, preobj_y_max, preobj_cls = pre_obj[1:]
                xycls_dis[num, pre_num] = (obj_x_min - preobj_x_min) + \
                                          (obj_y_min - preobj_y_min) + \
                                          (obj_x_max - preobj_x_max) + \
                                          (obj_y_max - preobj_y_max) + \
                                          (1,0)[obj_cls==preobj_cls]

        return match_list

if __name__ == '__main__':
    pass
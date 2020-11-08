# -*- coding: utf-8 -*-

import numpy as np
import json
from argparse import ArgumentParser


def get_intersections(coastline_points, validate_line):
    validate_diff = validate_line[1]-validate_line[0]
    norm = np.sqrt((validate_diff**2).sum())
    points_diff = coastline_points - validate_line[0]
    t = (points_diff*validate_diff).sum(axis=1)/norm**2
    dist = (np.matmul(points_diff, np.array([[0, -1], [1, 0]]))*validate_diff).sum(axis=1)/norm
    valid_positive_mask = (t >= 0)*(t <= 1)*(dist >= 0)
    valid_negative_mask = (t >= 0)*(t <= 1)*(dist < 0)
    positives = np.array([t[valid_positive_mask],
                          dist[valid_positive_mask]]).T
    negatives = np.array([t[valid_negative_mask],
                          dist[valid_negative_mask]]).T
    if len(positives) and len(negatives):
        positive_points = np.concatenate([positives, coastline_points[valid_positive_mask]],
                                         axis=1)
        negative_points = np.concatenate([negatives, coastline_points[valid_negative_mask]],
                                         axis=1)
        positive_mask = positive_points[:, 1] <= 1
        positive_near = positive_points[positive_mask]
        negative_mask = negative_points[:, 1] >= -1
        negative_near = negative_points[negative_mask]

        intersections = []
        if len(positive_near):
            intersections.append(validate_line[0] + validate_diff*positive_near[:, :1])
        if len(negative_near):
            intersections.append(validate_line[0] + validate_diff*negative_near[:, :1])

        if len(intersections):
            intersections = np.concatenate(intersections)
            
            return 0, intersections
        else:
            positive_nearest = positive_points[np.argmin(positive_points, axis=0)[1]]
            negative_nearest = negative_points[np.argmax(negative_points, axis=0)[1]]
            r = np.abs(negative_nearest[1])/(np.abs(negative_nearest[1])+positive_nearest[1])
            intersection = r*positive_nearest[-2:]+(1-r)*negative_nearest[-2:]
            
            return 1, intersection
    else:
        return 2, None


def ExtractError(sub, ans, options):
    eval_files = set(sub).intersection(set(ans))
    score = 0
    miss_cost = options['miss_cost']
    for eval_file in sorted(eval_files):
        validate_lines = np.array(ans[eval_file]['validate_lines'])
        ans_points = []
        valid_validate_lines = []
        for validate_line in validate_lines:
            flag, point = get_intersections(np.array(ans[eval_file]['coastline_points']), validate_line)
            if flag==0:
                ans_points.append(point.mean(axis=0))
                valid_validate_lines.append(validate_line)
            elif flag==1:
                ans_points.append(point)
                valid_validate_lines.append(validate_line)
        ans_points = np.array(ans_points)
        valid_validate_lines = np.array(valid_validate_lines)
        sub_points = np.array(sub[eval_file])
        err = 0
        for ans_point, validate_line in zip(ans_points, valid_validate_lines):
            flag, point = get_intersections(sub_points, validate_line)
            if flag==0:
                e = np.sqrt(((ans_point-point)**2).sum(axis=1).max())
                err += e
            elif flag==1:
                e = np.sqrt(((ans_point-point)**2).sum())
                err += e
            else:
                err += miss_cost
        error = err/len(ans_points)
        print(eval_file, 'Extract Error: {}'.format(error))
        score += error

    return score/len(eval_files)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--prediction-file', default = 'predictions.json')
    parser.add_argument('--annotation-file', default = 'annotations.json')

    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.prediction_file) as f:
        sub = json.load(f)

    with open(args.annotation_file) as f:
        ans = json.load(f)

    
    options = {'miss_cost': 100}
    
    score = ExtractError(sub, ans, options)
    print('\nOverall:', score)


if __name__ == '__main__':
    main()

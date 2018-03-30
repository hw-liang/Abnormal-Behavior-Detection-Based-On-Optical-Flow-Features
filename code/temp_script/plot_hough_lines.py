import cv2
import numpy as np
import os

def hough_lines(img):
    return cv2.HoughLines(cv2.Canny(img,105,130),1,np.pi/180,89)

def plot_hough_lines(lines,img):
    if lines is not None:
        lines1 = lines[:, 0, :]
        for rho, theta in lines1[:]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    else:
        print ('Hough line not find')
    return img;

def check_hough_lines():
    if not os.path.isdir('../ref_data/pics_with_hough_lines'):
        return False
    else:
        print ('pics with hough lines already exits')
        return False if input('If recalculation needed, please enter \'y\',that would induce overwritten. Else enter \'n\':\n')\
                       == 'y' else True

def load_and_show():
    print ('loading img...')
    for i in range(1, 201):
        new_img = cv2.imread('../ref_data/pics_with_hough_lines/pics_lines_' + str(i).zfill(3) + '.tif')
        cv2.imshow('img', new_img)
        # press ESC to exit
        if cv2.waitKey(200) & 0xff == 27:
            break;

def main(imgShowSkip=False):
    if not check_hough_lines():
        print ('calculating hough lines...')
        for i in range(1,201):
            img = cv2.imread('../original_pics/'+str(i).zfill(3)+'.tif')
            lines = hough_lines(img)
            new_img = plot_hough_lines(lines,img)
            hough_lines_only = plot_hough_lines(lines,np.zeros((img.shape[0],img.shape[1],3)))
            if not os.path.exists('../ref_data/pics_with_hough_lines'):
                os.mkdir('../ref_data/pics_with_hough_lines')
            cv2.imwrite('../ref_data/pics_with_hough_lines/pics_lines_' + str(i).zfill(3) + '.tif', new_img)
            if not os.path.exists('../ref_data/hough_lines_only'):
                os.mkdir('../ref_data/hough_lines_only')
            cv2.imwrite('../ref_data/hough_lines_only/lines_only_' + str(i).zfill(3) + '.tif', hough_lines_only)
            if not imgShowSkip:
                cv2.imshow('img', new_img)
                # press ESC to exit
                if cv2.waitKey(200) & 0xff == 27:
                    break;
    else:
        if not imgShowSkip:
            load_and_show()
    cv2.destroyAllWindows()

hough_main=main

if __name__ == '__main__':
    main()
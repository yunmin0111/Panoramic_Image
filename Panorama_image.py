import numpy as np
import os
import random
import cv2


# 이미지 로드용 함수
def load_images(image_paths):
    images = []
    for path in image_paths:
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                images.append(img)
            else:
                print(f"Error loading image: {path}")
        else:
            print(f"File not found: {path}")
    return images


# 가우시안 필터 함수
def gaussian_filter(image, kernel_size=5, sigma=1.0):

    k = kernel_size // 2
    x, y = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1))
    gaussian_kernel = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    gaussian_kernel /= gaussian_kernel.sum()  # 정규화

    padded_image = np.pad(image, ((k, k), (k, k)), mode='reflect')
    filtered_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            filtered_image[i, j] = np.sum(region * gaussian_kernel)
    
    return filtered_image

# Sobel 필터 함수
def sobel_filter(image, axis=None):

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    k = 1  
    padded_image = np.pad(image, ((k, k), (k, k)), mode='reflect')

    grad_x = np.zeros_like(image, dtype=np.float32)
    grad_y = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + 3, j:j + 3]
            grad_x[i, j] = np.sum(region * sobel_x)
            grad_y[i, j] = np.sum(region * sobel_y)

    if axis == 'x':
        return grad_x
    elif axis == 'y':
        return grad_y
    else:

        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return np.clip(gradient_magnitude, 0, 255).astype(np.uint8)

# Harris Coner Detection-코너 포인트 찾기 함수
def harris_corner_detection(image, block_size=3, k=0.04, threshold=0.1):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)


    Ix = sobel_filter(gray, axis='x')
    Iy = sobel_filter(gray, axis='y')

    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy
    
    Ixx = gaussian_filter(Ixx, kernel_size=block_size)
    Iyy = gaussian_filter(Iyy, kernel_size=block_size)
    Ixy = gaussian_filter(Ixy, kernel_size=block_size)

    height, width = gray.shape
    R = np.zeros_like(gray, dtype=np.float64)

    for y in range(block_size, height - block_size):
        for x in range(block_size, width - block_size):
           
            Sxx = np.sum(Ixx[y - block_size:y + block_size + 1, x - block_size:x + block_size + 1])
            Syy = np.sum(Iyy[y - block_size:y + block_size + 1, x - block_size:x + block_size + 1])
            Sxy = np.sum(Ixy[y - block_size:y + block_size + 1, x - block_size:x + block_size + 1])

           
            M = np.array([[Sxx, Sxy], [Sxy, Syy]])
            det_M = np.linalg.det(M)
            trace_M = np.trace(M)
            R[y, x] = det_M - k * (trace_M ** 2)

    R_normalized = (R - R.min()) / (R.max() - R.min())
    corners = np.argwhere(R_normalized > threshold)

    return corners

# RANSAC 함수-point matching
def ransac(points1, points2, threshold=3, iterations=1000):
   
    best_inliers = []
    best_matrix = None

    num_points = min(len(points1), len(points2))

    for _ in range(iterations):
 
        sample_indices = random.sample(range(num_points), 4)
        sampled_points1 = points1[sample_indices]
        sampled_points2 = points2[sample_indices]

        matrix = compute_homography(sampled_points1, sampled_points2)

        inliers = []
        for i, (p1, p2) in enumerate(zip(points1, points2)):
            transformed_p1 = transform_point(p1, matrix)
            distance = np.linalg.norm(transformed_p1 - p2)
            if distance < threshold:
                inliers.append(i)


        if len(inliers) > 0.5 * len(points1):
            return matrix, inliers 

 
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_matrix = matrix

    return best_matrix, best_inliers

# Homography 행렬 계산 함수
def compute_homography(points1, points2):
 
    A = []
    for p1, p2 in zip(points1, points2):
        x1, y1 = p1
        x2, y2 = p2
        A.append([-x1, -y1, -1, 0, 0, 0, x2 * x1, x2 * y1, x2])
        A.append([0, 0, 0, -x1, -y1, -1, y2 * x1, y2 * y1, y2])

    A = np.array(A)
    _, _, VT = np.linalg.svd(A)
    h = VT[-1]
    homography = h.reshape(3, 3)

    return homography

def transform_point(point, homography):
    x, y = point
    x_new = homography[0, 0] * x + homography[0, 1] * y + homography[0, 2]
    y_new = homography[1, 0] * x + homography[1, 1] * y + homography[1, 2]
    w = homography[2, 0] * x + homography[2, 1] * y + homography[2, 2]
    if w != 0:
        return x_new / w, y_new / w
    return x, y  

def preprocess_images(image_paths, target_size=(800, 600)):  
    images = load_images(image_paths)
    resized_images = [cv2.resize(img, target_size) for img in images]  
    return resized_images


# 이미지 겹치기 
def blend_images(result, img2, homography):
    h, w = img2.shape[:2]

    max_x = max(result.shape[1], w)
    max_y = max(result.shape[0], h)

    new_result = np.zeros((max_y, max_x, 3), dtype=np.uint8)

    for y in range(result.shape[0]):
        for x in range(result.shape[1]):
            new_result[y, x] = result[y, x]
    
    for y in range(h):
        for x in range(w):
            new_x, new_y = transform_point((x, y), homography)
            new_x = int(new_x)
            new_y = int(new_y)

            if 0 <= new_x < new_result.shape[1] and 0 <= new_y < new_result.shape[0]:
                new_result[new_y, new_x] = img2[y, x]

    return new_result

# 스티칭 함수
def stitch_images(images):
    result = images[0]

    for i in range(1, len(images)):
        img2 = images[i]

        corners1 = harris_corner_detection(result)
        corners2 = harris_corner_detection(img2)

        matched_points1 = corners1[:min(len(corners1), 10)]  
        matched_points2 = corners2[:min(len(corners2), 10)]  

        homography, _ = ransac(matched_points1, matched_points2)

        result = blend_images(result, img2, homography)

    return result


if __name__ == "__main__":

    image_paths = [

    "images\\testimg1.jpg",
    "images\\testimg2.jpg",
    "images\\testimg3.jpg",
    "images\\testimg4.jpg",
    "images\\testimg5.jpg",
    "images\\testimg6.jpg",
    "images\\testimg7.jpg",
    "images\\testimg8.jpg",
    "images\\testimg9.jpg",
    "images\\testimg10.jpg",
    ]

    preprocessed_images = preprocess_images(image_paths)

    stitched_image = stitch_images(preprocessed_images)


    cv2.imwrite('result.jpg', stitched_image)
    cv2.imshow('Stitched Image', stitched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

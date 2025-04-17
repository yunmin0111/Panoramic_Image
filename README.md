# Panoramic_Image

## Computer Vision Project

### 프로젝트 설명
#### : 이 프로젝트는 여러 장의 이미지를 이어 panorama image를 만드는 것을 목표로 한다. 주요 기능은 corner point를 정확하게 찾는 것, Point Matching을 함으로써 이미지를 적절하게 이어 붙이는 것 등이 있다. corner는 시점이 바뀌더라도 큰 변화가 없는 특징적인 point로, 물체의 중요한 특징을 잘 나타내는 feature pointer로 쓰일 수 있기 때문에 panorama 이미지에서는 이를 활용해 stitching을 한다. 이 과제에서는 image에 대한 이미지 전처리, 노이즈 제거를 한 후, corner point를 찾는다. 찾은 corner point를 바탕으로 Point를 Matching하고, RANSAC을 통해 최적의 Matching Point를 찾아낸다. Homography를 계산해 이미지에 대응되는 point를 연결하는 행렬을 얻고, Stitching으로 이미지를 붙인다. 이 후, 이미지의 더 나은 품질을 위해 Group Adjustment, Tone Mapping을 추가한다. 

### 프로젝트는 이미지 전처리, 노이즈 제거, 코너 포인트 찾기, Point Matching, RANSAC, Homography 계산, Group Adjust, Stitching(Tone Mapping) 순으로 파노라마 이미지를 만든다. 

### origin imagea: 활용한 이미지는 origin image 폴더에 있으며, 차이나타운 정문의 이미지를 직접 촬영해 사용했다.
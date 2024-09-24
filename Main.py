import tensorflow as tf

# model = tf.keras.models.load_model('C:\Users\Asus\Downloads\Model detect Number Plate\Ver 1\license_plate_model.h5')
model = tf.keras.models.load_model("object_detectionver_1.h5")


path = 'thay đường dẫn ảnh'
image = load_img(path) # PIL object
image = np.array(image,dtype=np.uint8) # 8 bit array (0,255)
image1 = load_img(path,target_size=(224,224))
image_arr_224 = img_to_array(image1)/255.0  # Convert into array and get the normalized output
h,w,d = image.shape

fig = px.imshow(image)
fig.update_layout(width=700, height=500,  margin=dict(l=10, r=10, b=10, t=10), xaxis_title='Image')
# reshape để phù hợp với đầu vào model
test_arr = image_arr_224.reshape(1,224,224,3)
# sử dụng model để dự đoán tọa độ bouding box của biển số xe
coords = model.predict(test_arr)

denorm = np.array([w,w,h,h])
coords = coords * denorm
coords = coords.astype(np.int32)
xmin, xmax,ymin,ymax = coords[0]
pt1 =(xmin,ymin)
pt2 =(xmax,ymax)
# in ra ảnh đã được khoanh vùng biển số
cv2.rectangle(image,pt1,pt2,(0,255,0),3)
fig = px.imshow(image)
fig.update_layout(width=700, height=500, margin=dict(l=10, r=10, b=10, t=10))
function visualize_box(image, box)
if size(image,3)~=0
imshow(image(box(3):box(4),box(1):box(2),:));
else
imshow(image(box(3):box(4),box(1):box(2)));
end
end

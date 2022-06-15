train_set_x_orig = hdf5read('train_catvnoncat.h5', 'train_set_x');
train_set_y_orig = hdf5read('train_catvnoncat.h5', 'train_set_y');

test_set_x_orig = hdf5read('test_catvnoncat.h5', 'test_set_x');
test_set_y_orig = hdf5read('test_catvnoncat.h5', 'test_set_y');

classes = hdf5read('test_catvnoncat.h5', 'list_classes');

m_train = size(train_set_x_orig, 4);
m_test = size(test_set_x_orig, 4);
num_px = size(train_set_x_orig, 2);
num_channel = size(train_set_x_orig, 1);

fprintf('Dataset dimensions: \n');
fprintf('Number of training examples: m_train = %d \n', m_train);
fprintf('Number of testing examples: m_test  = %d \n', m_test);
fprintf('Height/Width of each image: num_px  = %d \n', num_px);
fprintf('Each image is of size:       (%d, %d) \n', num_px, num_px);
fprintf('train_set_x shape: (%s) \n', num2str(size(train_set_x_orig)));
fprintf('train_set_y shape: (%s) \n', num2str(size(train_set_y_orig)));

% If you want to display an image, uncomment the following code
% img = train_set_x_orig(:, :, :, 3);
% img = permute(img, [2 3 1]);
% figure();
% imshow(img);
function A = get_atmosphere(image,dark_channel)
[m,n,~] = size(image);
n_pixels = m*n;
n_search_pixels = floor(n_pixels*0.001);
dark_vec = reshape(dark_channel,[n_pixels,1]);
image_vec = reshape(image,[n_pixels,3]);
[~,indices] = sort(dark_vec,'descend');
accumulator = zeros(1,3);
for k = 1:n_search_pixels
	accumulator = accumulator + image_vec(indices(k),:);
end
A = zeros(1,1,3);
for k = 1:3
	A(1,1,k) = accumulator(1,k)/n_search_pixels;
end

end
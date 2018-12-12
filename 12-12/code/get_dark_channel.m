function dark_channel = get_dark_channel(image,win_size)
[m,n,~] = size(image);
pad_size = floor(win_size/2);
padded_image = padarray(image,[pad_size pad_size],Inf);
dark_channel = zeros(m,n);
for i = 1:m
	for j = 1:n
		patch = padded_image(i:(i+win_size-1),j:(j+win_size-1),:);
		dark_channel(i,j) = min(patch(:));
	end
end
end
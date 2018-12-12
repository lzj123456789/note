clc;
clear ;
tic
in = double(imread('IMG_7360.JPG'))/(2^8-1);
in = imresize(in,0.5);
I = 1-in;
dark_channel = get_dark_channel(I,15);
A = get_atmosphere(I,dark_channel);
Y = 0.299 * I(:,:,1) + 0.587 *I(:,:,2) + 0.114*I(:,:,3);
t = 1-0.98*Y;
%t = medfilt2(t,[15,15]);
[m,n,~] = size(in);
rep_A = repmat(A,[m,n,1]);
rep_t = repmat(max(t,0.01),[1,1,3]);

J = (I- rep_A)./rep_t + rep_A;
J = 1-J;
toc

subplot(2,2,1);
imshow(uint8(in*(2^8-1)));
subplot(2,2,2);
imshow(uint8(I*(2^8-1)));
subplot(2,2,3);
imshow(uint8(Y*(2^8-1)));
subplot(2,2,4);
imshow(uint8(J*(2^8-1)));
imwrite(J,'test.jpg')

clc;
clear ;
tic
in = double(imread('Img3.tif'))/(2^16-1);
I = 1-in;
dark_channel = get_dark_channel(I,15);
A = get_atmosphere(I,dark_channel);
Y = 0.299 * I(:,:,1) + 0.587 *I(:,:,2) + 0.114*I(:,:,3);
t = 1-0.98*Y;
t = medfilt2(t,[7,7]);
[m,n,~] = size(in);
rep_A = repmat(A,[m,n,1]);
rep_t = repmat(max(t,0.01),[1,1,3]);

J = (I- rep_A)./rep_t + rep_A;
J = 1-J;
toc

subplot(2,2,1);
imshow(uint16(in*(2^16-1)));
subplot(2,2,2);
imshow(uint16(I*(2^16-1)));
subplot(2,2,3);
imshow(uint16(Y*(2^16-1)));
subplot(2,2,4);
imshow(uint16(J*(2^16-1)));
imwrite(J,'test.jpg')


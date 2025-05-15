
function [S,Q] = muGIF(T,R,alpha_t,alpha_r,maxiter,mode)  
%--------------------------------------------------------------------------
%A demo code of "Mutually Guided Image Filtering" by Xiaojie Guo
%
%Input: T [target image T_0 in the paper]
%       R [reference image R_0 in the paper]
%       alpha_t [alpha_t in the paper]
%       alpha_r [alpha_r in the paper]
%       maxiter [max number of iterations]
%       mode [0: dynamic/dynamic; 1: static/dynamic; 2: dynamic only]
%Output: S [processed T]
%        Q [processed R]
%--------------------------------------------------------------------------

    T0 = im2double(T);
    R0 = im2double(R);
    t = T0;
    r = R0;
    epsr = 0.01;
    epst = 0.01;

   if mode ==0    % dynamic/dynamic
        for i = 1 :maxiter
            [wrx, wry] = computeTextureWeights(r,epsr);
            [wtx, wty] = computeTextureWeights(t,epst);
            wx = wtx.*wrx;
            wy = wty.*wry;
            t = solveLinearEquation(T0, wx, wy, alpha_t);
            r = solveLinearEquation(R0, wx, wy, alpha_r);
        end
   end
   if mode == 1   % static/dynamic
        [wrx, wry] = computeTextureWeights(r,epsr);
        for i = 1 :maxiter
            [wtx, wty] = computeTextureWeights(t,epst);
            wx = wtx.*wrx;
            wy = wty.*wry;
            t = solveLinearEquation(T0, wx, wy, alpha_t);
        end
   end
   if mode == 2   % dynamic only
        for i = 1 :maxiter
            [wtx, wty] = computeTextureWeights(t,epst);
            wx = wtx.^2;  %x方向的权重系数
            wy = wty.^2;
            t = solveLinearEquation(T0, wx, wy, alpha_t);
        end
   end
    S = t;   
    Q = r;
end


function [retx, rety] = computeTextureWeights(fin,vareps_s)
   fx = diff(fin,1,2);
   fx = padarray(fx, [0 1], 'post');
   fy = diff(fin,1,1);
   fy = padarray(fy, [1 0], 'post');  %差分

   retx = max(max(abs(fx),[],3),vareps_s).^(-1); 
   rety = max(max(abs(fy),[],3),vareps_s).^(-1);  %分母，根据设定的梯度阈值分类

   retx(:,end) = 0;
   rety(end,:) = 0;
end



function OUT = solveLinearEquation(IN, wx, wy, lambda)
% WLS
    [r,c,ch] = size(IN);
    k = r*c;
    dx = -lambda*wx(:);
    dy = -lambda*wy(:);
    B(:,1) = dx;
    B(:,2) = dy;
    d = [-r,-1];           %生成的稀疏矩阵A在主对角线的下三角的-1,和-100有值，从头开始排，-1有1000*1000-1个数，-100有
    A = spdiags(B,d,k,k);  %A = spdiags（B，D ,m, n），产生一个m×n稀疏矩阵A，其元素是B中的列元素放在由D指定的对角线位置上。
    e = dx;
    w = padarray(dx, r, 'pre');w = w(1:end-r);  %把0的数值移到有具体数值的前面
    s = dy;
    n = padarray(dy, 1, 'pre'); n = n(1:end-1);
    D = 1-(e+w+s+n);
    A = A + A' + spdiags(D, 0, k, k); 
    L = ichol(A,struct('michol','on'));  %Cholesky分解将矩阵X分解成一个下三角矩阵和上三角矩阵的乘积  
    
     OUT = IN;
        for ii=1:ch
            tin = IN(:,:,ii);
            [tout,~] = pcg(A, tin(:),.01,max(min(ceil(lambda*100),40),10), L, L'); %容错为0.01
            OUT(:,:,ii) = reshape(tout, r, c);
        end    
end
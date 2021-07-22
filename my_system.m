function dy = my_system(t,y,a,w_ex,s,ji,in_signal,w0,w_e,n,K)
% n --- кол-во уравнений системы первого порядка

dy = zeros(n,1);
m = n*0.5;
dy(1:m) = y(m + 1:n);
% K = reshape(K, [m, m]);
for i = 1:m
    summa = 0;
    for j = 1:m
        summa = summa + K(i, j) * y(m + j);
    end
    dy(m + i) = -a*w_ex*y(m + i)+w_ex*s*(ji + in_signal.*sin(w0*t))-0.5*w_ex*w_e*sin(2*y(i))+w_ex*summa;
    % dy(m + i) = -a*w_ex*y(m + i)+w_ex*s*(ji)-0.5*w_ex*w_e*sin(2*y(i))+w_ex*summa;
end

end

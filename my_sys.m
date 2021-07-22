function dy = my_sys(t, y, n, Omega, K)
dy = zeros(n,1);
for i = 1:n
    summa = 0;
    for j = 1:n
        summa = summa + K(i, j) * cos(y(j));
    end
    dy(i) = Omega - sin(y(i)) * summa;
end
end


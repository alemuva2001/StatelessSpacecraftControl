function att_validation(tout, I, w)
%Check if attitude is correct: the angular momentum and kinetic energy preserve their value
N = length(tout);
h_norm = zeros(N,1);
T = zeros(N,1);

for i=1:N
    %Angular momentum conservation
    h = I*w(i,:)';
    h_norm(i) = norm(h);

    %Kinetic energy conservation
    T(i)= 1/2*w(i,:)*I*w(i,:)';
end

figure('Name','Attitude Validation');
subplot(2,1,1)
plot(tout, h_norm,'LineWidth',2);
grid on
grid minor
legend('Angular momentum')
subplot(2,1,2)
plot(tout, T,'LineWidth',2);
grid on
grid minor
legend('Kinetic energy')
end
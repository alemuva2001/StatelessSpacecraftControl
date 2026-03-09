function orb_validation(tout, r_N, r_Nd, mu_o)
%Check if the orbital dynamics are correct: specific energy and specific
%angular momentum are constants
N = length(tout);
h_o = zeros(N,1);
epsilon = zeros(N,1);

rx = r_N(:,1);
ry = r_N(:,2);
rz = r_N(:,3);

r_o = [rx,ry,rz];
r_od = [r_Nd(:,1), r_Nd(:,2), r_Nd(:,3)];

for i=1:N
    r_ins = r_o(i,:);
    v_ins = r_od(i,:);

    %Specific angular momentum conservation
    h_o(i) = norm(cross(r_ins,v_ins));
    
    %Specific energy conservation
    epsilon(i) = 1/2*dot(v_ins,v_ins) - mu_o/norm(r_ins);
end

figure('Name','Orbital validation')
subplot(2,1,1)
plot(tout, h_o, 'LineWidth',2);
grid on
grid minor
legend('Specific angular momentum')
subplot(2,1,2)
plot(tout, epsilon, 'LineWidth',2);
grid on
grid minor
legend('Specific energy')

end
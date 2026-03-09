function orbit_visualization(tout, r_N, Rt)
%Orbit visualization
rx = r_N(:,1);
ry = r_N(:,2);
rz = r_N(:,3);

figure('Name','Órbita');
plot3(rx, ry, rz, 'b-', 'LineWidth',2);
hold on
grid on
grid minor
legend('orbit trajectory')

%We also draw an sphere representing the Earth
draw_earth_3d(Rt * 1000);

legend('Orbit trajectory', 'Location', 'best');
view(3); % Vista 3D isométrica

end
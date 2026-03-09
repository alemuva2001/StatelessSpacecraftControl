function T_ext_representation(tout, T_tot, T_srp)
    
    figure('Name','External torques')
    subplot(2,1,1)
    plot(tout, T_tot', 'LineWidth',2)
    title('Total Torque')
    legend('T_{tot_x}', 'T_{tot_y}', 'T_{tot_z}')
    subplot(2,1,2)
    plot(tout, T_srp', 'LineWidth',2)
    title('SRP Torque')
    legend('T_{srp_x}', 'T_{srp_y}', 'T_{srp_z}')

end
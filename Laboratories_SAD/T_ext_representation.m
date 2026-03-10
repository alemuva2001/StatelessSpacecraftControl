function T_ext_representation(tout, T_tot, T_srp, T_gg, T_mag)
    
    figure('Name','External torques')
    subplot(1,2,1)
    plot(tout, T_tot',':', 'LineWidth',2)
    hold on
    plot(tout, vecnorm(T_tot, 2, 2)', 'LineWidth',2)
    title('Total Torque')
    grid on; grid minor;
    legend('T_{tot_x}', 'T_{tot_y}', 'T_{tot_z}','T_{tot}')

    subplot(1,2,2)
    semilogy(tout, vecnorm(T_srp, 2, 2)', 'LineWidth',2)
    title('SRP Torque')
    hold on
    subplot(1,2,2)
    semilogy(tout, vecnorm(T_gg, 2, 2)', 'LineWidth',2)
    title('GG Torque')
    subplot(1,2,2)
    semilogy(tout, vecnorm(T_mag, 2, 2)', 'LineWidth',2)
    title('External Torque (O)')
    legend('T_{SRP}', 'T_{gg}', 'T_{mag}')
    grid on; grid minor;

end
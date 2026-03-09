%% Draw Euler angles
function euler_visualization(tout, A)

    N = length(tout);
    roll = zeros(N,1);
    pitch = zeros(N,1);
    yaw = zeros(N,1);
    
    for i=1:N
        roll(i) = atan2(A(3,2,i),A(3,3,i));
        pitch(i) = atan2(-A(3,1,i),sqrt(A(3,2,i)^2+A(3,3,i)^2));
        yaw(i) = atan2(A(2,1,i),A(1,1,i));
    end
    
    figure('Name','Euler Angles')
    subplot(3,1,1)
    plot(tout,roll,'LineWidth',2)
    grid on
    grid minor
    legend('roll')
    
    subplot(3,1,2)
    plot(tout,pitch,'LineWidth',2)
    grid on
    grid minor
    legend('pitch')
    
    subplot(3,1,3)
    plot(tout,yaw,'LineWidth',2)
    grid on
    grid minor
    legend('yaw')
end
%% Shuttle - classical cascaded (geometric) controller in Pegasus: figures and metrics
% Data: data/episode.csv and data/metrics.json written by export_results.py from the Pegasus run of run.sh
% (goal of the example episodes of the reinforcement learning studies, 1.59 m; x and y relative to the spawn point and
% z as the altitude, as in those episodes). All text uses the LaTeX interpreter (Computer Modern).
%
%   plot_results               % all figures, saved to figures/ (PDF + PNG) (Response, Reference)
%   plot_results(false)        % draw only
%   plot_results(true, true)   % also the ideal model, dotted, for comparison
%
% The ideal model is a MATLAB simulation (rigid body, thrust and torques
% applied directly, semi-implicit Euler at 10 ms), run here for the same goal and gains. It is drawn dotted, so the
% effect of the rotor dynamics, allocation and saturation of Pegasus can be checked.
%
% Figures (file names):
%   position_response, velocity_response, control_inputs (T, tau_x, tau_y, tau_z), attitude_error tr(I - Rd^T R)
% Requires MATLAB R2020b or later.
function plot_results(saveFigures, showIdeal)
if nargin < 1, saveFigures = true; end
if nargin < 2, showIdeal = false; end
here = fileparts(mfilename('fullpath'));
dataDir = fullfile(here, 'data');
figDir  = fullfile(here, 'figures');
if saveFigures && ~exist(figDir, 'dir'), mkdir(figDir); end
names = {'defaultTextInterpreter', 'defaultAxesTickLabelInterpreter', 'defaultLegendInterpreter', ...
         'defaultAxesFontName', 'defaultTextFontName', 'defaultLegendFontName'};
previous = cellfun(@(n) get(groot, n), names, 'UniformOutput', false);
for k = 1:3, set(groot, names{k}, 'latex'); end
set(groot, 'defaultAxesFontName', 'Computer Modern', 'defaultTextFontName', 'Computer Modern', ...
           'defaultLegendFontName', 'Computer Modern');
restore = onCleanup(@() restoreDefaults(names, previous));
D = readtable(fullfile(dataDir, 'episode.csv'));
M = jsondecode(fileread(fullfile(dataDir, 'metrics.json')));
tEnd = M.duration_s;
m = D.t <= tEnd + 1e-9;
t = D.t(m)';
p = [D.x(m) D.y(m) D.z(m)]';
v = [D.vx(m) D.vy(m) D.vz(m)]';
goal = [D.goal_x(1); D.goal_y(1); D.goal_z(1)];
U = [D.T_cmd(m) D.tau_x(m) D.tau_y(m) D.tau_z(m)]';
eTr = D.att_err_tr(m)';
ideal = [];
if showIdeal
    ideal = simulateIdeal(goal - p(:, 1), tEnd);   % the ideal model starts at the origin
    ideal.p = ideal.p + p(:, 1);
end
S.font = 9;
S.response = [0.10 0.30 0.65];
S.ideal = [0.85 0.33 0.10];
% --- Position ---
f = newFigure(S);
tl = tiledlayout(f, 3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
plot3Axis(tl, t, p, goal .* ones(3, numel(t)), field(ideal, 'p'), {'$p_x$ [m]', '$p_y$ [m]', '$p_z$ [m]'}, S, 'northeast');
saveFig(f, 'position_response');
% --- Velocity ---
f = newFigure(S);
tl = tiledlayout(f, 3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
plot3Axis(tl, t, v, zeros(3, numel(t)), field(ideal, 'v'), {'$v_x$ [m/s]', '$v_y$ [m/s]', '$v_z$ [m/s]'}, S, 'southeast');
saveFig(f, 'velocity_response');
% --- Control inputs ---
f = newFigure(S);
tl = tiledlayout(f, 4, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
labels = {'$T$ [N]', '$\tau_x$ [N\,m]', '$\tau_y$ [N\,m]', '$\tau_z$ [N\,m]'};
for i = 1:4
    ax = nexttile(tl); hold(ax, 'on');
    h = stairs(ax, t, U(i, :), '-', 'Color', S.response, 'LineWidth', 1.4);
    if ~isempty(ideal)
        hi = stairs(ax, ideal.t, ideal.U(i, :), ':', 'Color', S.ideal, 'LineWidth', 1.4);
    end
    ylabel(ax, labels{i});
    xlim(ax, [0, tEnd]);
    formatAxis(ax, S);
    if i < 4
        ax.XTickLabel = [];
    else
        xlabel(ax, 'Time [s]');
        if ~isempty(ideal)
            legend(ax, [h hi], {'Pegasus', 'Ideal model'}, 'Location', 'best', 'Box', 'off');
        end
    end
end
saveFig(f, 'control_inputs');
% --- Attitude error ---
f = newFigure(S);
ax = axes(f); hold(ax, 'on');
h = plot(ax, t, eTr, '-', 'Color', S.response, 'LineWidth', 1.4);
if ~isempty(ideal)
    hi = plot(ax, ideal.t, ideal.eTr, ':', 'Color', S.ideal, 'LineWidth', 1.4);
    legend(ax, [h hi], {'Pegasus', 'Ideal model'}, 'Location', 'best', 'Box', 'off');
end
ylabel(ax, '$\mathrm{tr}(I - R_d^\top R)$');
xlabel(ax, 'Time [s]');
xlim(ax, [0, tEnd]);
formatAxis(ax, S);
saveFig(f, 'attitude_error');
% --- Metrics ---
fprintf('\nPegasus, first %.0f s (initial distance %.2f m):\n', tEnd, M.initial_distance_m);
ax = 'xyz';
for i = 1:3
    a = M.axes.(ax(i));
    fprintf('  %s: settling (2%%) %.2f s, overshoot %.1f %%, final error %.1f mm\n', ax(i), ...
            a.settling_time_2pct_s, a.overshoot_percent, 1e3 * a.final_error_m);
end
fprintf('  final error %.1f mm (%.1f mm at 10 s)\n', 1e3 * M.final_error_m, 1e3 * M.final_error_10s_m);
fprintf('  thrust %.1f to %.1f N (hover %.1f N, maximum %.0f N), rotor speed up to %.0f of %.0f rad/s\n', ...
        M.min_thrust_N, M.peak_thrust_N, M.hover_thrust_N, M.max_thrust_N, M.peak_rotor_speed_rad_s, ...
        M.max_rotor_speed_rad_s);
fprintf('  peak torques %.2f, %.2f, %.2f N m, peak tilt %.1f deg\n', M.peak_torque_Nm.x, M.peak_torque_Nm.y, ...
        M.peak_torque_Nm.z, M.peak_tilt_deg);
if ~isempty(ideal)
    err = ideal.p - goal;
    fprintf('Ideal model: final error %.1f mm, settling (2%%) x/y/z %.2f/%.2f/%.2f s, max |p - p_Pegasus| %.1f cm\n', ...
            1e3 * norm(err(:, end)), settling(ideal.t, err(1, :)), settling(ideal.t, err(2, :)), ...
            settling(ideal.t, err(3, :)), 100 * max(vecnorm(ideal.p - interp1(t, p', ideal.t)')));
end
    function saveFig(fig, name)
        % PNG with exportgraphics; PDF with print -painters (same as the other results/ scripts): the LaTeX text
        % becomes vector outlines instead of selectable text set in a substitute font.
        if ~saveFigures, return; end
        drawnow;
        exportgraphics(fig, fullfile(figDir, [name '.png']), 'Resolution', 300);
        set(fig, 'PaperUnits', 'centimeters');
        pos = get(fig, 'Position');
        set(fig, 'PaperSize', pos(3:4), 'PaperPosition', [0 0 pos(3:4)]);
        print(fig, fullfile(figDir, [name '.pdf']), '-dpdf', '-painters');
    end
end
%% --- Ideal model (MATLAB simulation) ---
function out = simulateIdeal(goal, tEnd)
% Rigid body with mass m and inertia J, thrust and torques applied directly (no rotor dynamics, allocation or
% saturation), semi-implicit Euler at dt = 10 ms, from rest and level at the origin. Same controller and gains.
m = 3.5; J = diag([0.026615, 0.026615, 0.048026]); g = [0; 0; 9.81]; dt = 0.01;
Kp = 3.15 * eye(3); Kv = 2.37 * eye(3); kR = 3.5; kw = 0.5; yaw = 0; e3 = [0; 0; 1];
t = 0:dt:tEnd; n = numel(t);
p = zeros(3, n); v = zeros(3, n); R = eye(3); w = zeros(3, 1);
U = zeros(4, n); eTr = zeros(1, n);
for k = 1:n
    a = g + Kp * (goal - p(:, k)) + Kv * (-v(:, k));            % u + g e3, with v_d = 0 and a_d = 0
    zd = a / norm(a);
    yc = [-sin(yaw); cos(yaw); 0];
    xd = cross(yc, zd); xd = xd / norm(xd);
    Rd = [xd, cross(zd, xd), zd];
    T = max(m * ((R * e3)' * a), 0);
    Re = Rd' * R;
    eR = 0.5 * [Re(3, 2) - Re(2, 3); Re(1, 3) - Re(3, 1); Re(2, 1) - Re(1, 2)];
    tau = -kR * eR - kw * w + cross(w, J * w);                % w_d = 0
    U(:, k) = [T; tau]; eTr(k) = trace(eye(3) - Rd' * R);
    if k == n, break; end
    w = w + dt * (J \ (tau - cross(w, J * w)));
    R = R * expm([0 -w(3) w(2); w(3) 0 -w(1); -w(2) w(1) 0] * dt);
    p(:, k + 1) = p(:, k) + dt * v(:, k);
    v(:, k + 1) = v(:, k) + dt * (-g + T * (R * e3) / m);
end
out = struct('t', t, 'p', p, 'v', v, 'U', U, 'eTr', eTr);
end
%% --- Helpers ---
function x = field(s, name)
if isempty(s), x = []; else, x = struct('t', s.t, 'y', s.(name)); end
end
function ts = settling(t, err)
outside = abs(err) > 0.02 * abs(err(1));
last = find(outside, 1, 'last');
if isempty(last), ts = 0; elseif last < numel(t), ts = t(last + 1); else, ts = NaN; end
end
function plot3Axis(tl, t, sig, ref, ideal, ylabels, S, legendLocation)
for i = 1:3
    ax = nexttile(tl); hold(ax, 'on');
    h1 = plot(ax, t, sig(i, :), '-', 'Color', S.response, 'LineWidth', 1.4);
    h2 = plot(ax, t, ref(i, :), '--', 'Color', 'k', 'LineWidth', 1.1);
    values = [sig(i, :), ref(i, :)];
    if ~isempty(ideal)
        h3 = plot(ax, ideal.t, ideal.y(i, :), ':', 'Color', S.ideal, 'LineWidth', 1.4);
        values = [values, ideal.y(i, :)]; %#ok<AGROW>
    end
    span = max(max(values) - min(values), 1e-3);
    ylim(ax, [min(values) - 0.08 * span, max(values) + 0.08 * span]);
    xlim(ax, [t(1), t(end)]);
    ylabel(ax, ylabels{i});
    formatAxis(ax, S);
    if i == 1
        if isempty(ideal)
            legend(ax, [h1 h2], {'Response', 'Reference'}, 'Location', legendLocation, 'Box', 'off');
        else
            legend(ax, [h1 h3 h2], {'Pegasus', 'Ideal model', 'Reference'}, 'Location', legendLocation, 'Box', 'off');
        end
    end
    if i < 3
        ax.XTickLabel = [];
    else
        xlabel(ax, 'Time [s]');
    end
end
end
function f = newFigure(S)
f = figure('Units', 'centimeters', 'Color', 'w', 'WindowStyle', 'normal');
f.Position = [2 2 15 10];
set(f, 'DefaultAxesFontSize', S.font);
end
function formatAxis(ax, S)
set(ax, 'FontName', 'Computer Modern', 'FontSize', S.font, 'Box', 'on', 'TickDir', 'in', 'LineWidth', 0.6);
grid(ax, 'on'); ax.GridAlpha = 0.15;
end
function restoreDefaults(names, previous)
for k = 1:numel(names), set(groot, names{k}, previous{k}); end
end
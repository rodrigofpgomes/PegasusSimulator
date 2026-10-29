%% Shuttle - Direct Thrust and Moment Control: figures of the final study (10 seeds)
% Data: data/*.csv written by export_results.py (study_10seeds, k_T = 1.9; study_10seeds_kt3p9, k_T = 3.9).
% All text uses the LaTeX interpreter, i.e. Computer Modern.
% Statistics over seeds: median and interquartile range (robust to the single failed training).
%
%   plot_results                % all figures, saved to figures/ (PDF + PNG)
%   plot_results(false)         % draw only
%   plot_results(true, 9, 4)    % seed of the reference episode (9), seed of the yaw-penalty episode (4)
%
% Figures, in the order of the text:
%   fig01 training progress          fig02 mean error vs time         fig03 RMSE and t20 vs c_M
%   fig04-07 reference moment scale (c_M 0.25, c_D 0.8): position, velocity, angular rate, input
%   fig08 mean yaw rate of each policy                                fig09 angular rate vs yaw weight
%   fig10 yaw-penalty episode        fig11 thrust action for k_T = 1.9, 3.9 and 3.9 hover-centred
% Requires MATLAB R2020b or later.

function plot_results(saveFigures, bestSeed, yawSeed)
if nargin < 1, saveFigures = true; end
if nargin < 2, bestSeed = 9; end      % median final error among the ten seeds of c_M 0.25, c_D 0.8
ktSeed = 5;                           % fig11: median final error among the ten seeds of c_M 0.25, c_D 0.8, w_psi 1.01
if nargin < 3, yawSeed = 4; end       % c_M 0.25: rotates with w_psi 0.01, partly with 0.11, not with 1.01
here = fileparts(mfilename('fullpath'));
dataDir = fullfile(here, 'data');
figDir  = fullfile(here, 'figures');
if saveFigures && ~exist(figDir, 'dir'), mkdir(figDir); end

% Computer Modern: every text object through the LaTeX interpreter
names = {'defaultTextInterpreter', 'defaultAxesTickLabelInterpreter', 'defaultLegendInterpreter', ...
         'defaultColorbarTickLabelInterpreter', 'defaultAxesFontName', 'defaultTextFontName', ...
         'defaultLegendFontName', 'defaultAxesFontSize', 'defaultTextFontSize', 'defaultLegendFontSize'};
previous = cellfun(@(n) get(groot, n), names, 'UniformOutput', false);
for k = 1:4, set(groot, names{k}, 'latex'); end      % the first four are interpreters
set(groot, 'defaultAxesFontName', 'Computer Modern', 'defaultTextFontName', 'Computer Modern', ...
           'defaultLegendFontName', 'Computer Modern', 'defaultAxesFontSize', 9, 'defaultTextFontSize', 9, ...
           'defaultLegendFontSize', 9);
restore = onCleanup(@() restoreDefaults(names, previous));

M  = readtable(fullfile(dataDir, 'run_metrics.csv'), 'TextType', 'string');
C  = readtable(fullfile(dataDir, 'eval_curves.csv'), 'TextType', 'string');
T  = readtable(fullfile(dataDir, 'training_curves.csv'), 'TextType', 'string');
EP = readtable(fullfile(dataDir, 'example_episode.csv'), 'TextType', 'string');

S = style();
scales = [0.01 0.06 0.12 0.25 0.5 1.0];
dens   = [0.8 1.5];
G  = M(M.thrust_to_weight == 1.9 & M.w_psi < 0.02, :);       % moment-scale x denominator grid
scaleNames = compose('$c_M = %.2f$ N\\,m', scales);

%% fig01 - training progress (final distance of the training episodes, stochastic policy)
[f, tl] = twoPanels(S);
h = gobjects(0);
for d = 1:2
    ax = nexttile(tl); hold(ax, 'on');
    for m = 1:numel(scales)
        sel = T.thrust_to_weight == 1.9 & T.w_psi < 0.02 & T.moment_scale == scales(m) & T.denominator == dens(d);
        [x, Y] = bySeed(T(sel, :), 'iteration', 'final_distance_m', (1:200)', 5);
        h(m) = medianBand(ax, x, Y, S.scaleColor(m, :));
    end
    set(ax, 'YScale', 'log'); xlim(ax, [0 200]); ylim(ax, [0.01 5]);
    xlabel(ax, 'PPO iteration'); panelTitle(ax, d, dens(d));
    if d == 1, ylabel(ax, 'Final position error $\|\mathbf{e}_p\|$ [m]'); end
    formatAxis(ax, S);
end
bottomLegend(h, scaleNames, 3);
saveFigure(f, figDir, 'fig01_training_progress', saveFigures);

%% fig02 - mean position error over the 200 goals versus time
[f, tl] = twoPanels(S);
h = gobjects(0);
for d = 1:2
    ax = nexttile(tl); hold(ax, 'on');
    for m = 1:numel(scales)
        sel = C.w_psi < 0.02 & C.moment_scale == scales(m) & C.denominator == dens(d);
        [x, Y] = bySeed(C(sel, :), 't', 'err_mean', (0.05:0.1:9.95)', 1);
        h(m) = medianBand(ax, x, Y, S.scaleColor(m, :));
    end
    set(ax, 'YScale', 'log'); xlim(ax, [0 10]); ylim(ax, [0.01 2.5]);
    xlabel(ax, 'Time [s]'); panelTitle(ax, d, dens(d));
    if d == 1, ylabel(ax, 'Mean position error $\|\mathbf{e}_p\|$ [m]'); end
    formatAxis(ax, S);
end
bottomLegend(h, scaleNames, 3);
saveFigure(f, figDir, 'fig02_error_vs_time', saveFigures);

%% fig03 - RMSE and convergence time versus moment scale (one marker per seed)
[f, tl] = twoPanels(S);
fields = {'rmse_m', 't20_s'};
ylabs  = {'Position RMSE [m]', 'Convergence time $t_{20}$ [s]'};
ttls   = {'(a) RMSE', '(b) Convergence time'};
for k = 1:2
    ax = nexttile(tl); hold(ax, 'on');
    h = gobjects(0);
    for d = 1:2
        for m = 1:numel(scales)
            v = G.(fields{k})(G.moment_scale == scales(m) & G.denominator == dens(d));
            hh = seedStrip(ax, m + S.denOffset(d), v, S.denColor(d, :), S.denMarker{d});
            if m == 1, h(d) = hh; end
        end
    end
    formatCategoryAxis(ax, S, scales, 'Moment scale $c_M$ [N\,m]');
    ylabel(ax, ylabs{k}); title(ax, ttls{k}, 'FontWeight', 'normal');
end
bottomLegend(h, {'$c_D = 0.8$ m', '$c_D = 1.5$ m'});
saveFigure(f, figDir, 'fig03_rmse_t20_vs_moment_scale', saveFigures);

%% fig04-fig07 - reference moment scale (c_M 0.25, c_D 0.8, w_psi 0.01): one goal at the median initial distance
E = EP(EP.thrust_to_weight == 1.9 & EP.experiment == "ms025_den08" & EP.seed == bestSeed, :);
E.zero = zeros(height(E), 1);
E.hover = E.weight_N;
cM = E.moment_scale(1);
episodeFigure(S, E, {'x', 'y', 'z'}, {'goal_x', 'goal_y', 'goal_z'}, ...
    {'$p_x$ [m]', '$p_y$ [m]', '$p_z$ [m]'}, {'Position', 'Reference'}, {}, ...
    figDir, 'fig04_best_position', saveFigures);
episodeFigure(S, E, {'vx', 'vy', 'vz'}, {'zero', 'zero', 'zero'}, ...
    {'$v_x$ [m/s]', '$v_y$ [m/s]', '$v_z$ [m/s]'}, {'Velocity', 'Reference ($\mathbf{v} = 0$)'}, {}, ...
    figDir, 'fig05_best_velocity', saveFigures);
episodeFigure(S, E, {'wx', 'wy', 'wz'}, {'zero', 'zero', 'zero'}, ...
    {'$\omega_x$ [rad/s]', '$\omega_y$ [rad/s]', '$\omega_z$ [rad/s]'}, ...
    {'Angular rate', 'Desired ($\omega = 0$)'}, {}, figDir, 'fig06_best_angular_rate', saveFigures);
episodeFigure(S, E, {'thrust_N', 'tau_x_Nm', 'tau_y_Nm', 'tau_z_Nm'}, {'hover', '', '', ''}, ...
    {'$T$ [N]', '$\tau_x$ [N\,m]', '$\tau_y$ [N\,m]', '$\tau_z$ [N\,m]'}, ...
    {'Applied input', 'Hover thrust'}, ...
    {[0, 1.9 * E.weight_N(1)], [-cM, cM], [-cM, cM], [-cM, cM]}, figDir, 'fig07_best_input', saveFigures);

%% fig08 - mean yaw rate of each of the 120 policies of the grid
f = newFigure(S, 8, 6.5);
tl = tiledlayout(f, 1, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
ax = nexttile(tl); hold(ax, 'on');
h = gobjects(0);
for d = 1:2
    for m = 1:numel(scales)
        v = G.wz_mean_signed(G.moment_scale == scales(m) & G.denominator == dens(d));
        x = m + S.denOffset(d) + linspace(-0.07, 0.07, numel(v))';
        hh = scatter(ax, x, v, 22, S.denColor(d, :), S.denMarker{d}, 'filled');
        if m == 1, h(d) = hh; end
    end
end
yline(ax, 0, '-', 'Color', [0.3 0.3 0.3], 'HandleVisibility', 'off');
yline(ax, 0.5, ':', 'Color', [0.3 0.3 0.3], 'HandleVisibility', 'off');
yline(ax, -0.5, ':', 'Color', [0.3 0.3 0.3], 'HandleVisibility', 'off');
formatCategoryAxis(ax, S, scales, 'Moment scale $c_M$ [N\,m]');
ylabel(ax, 'Mean yaw rate [rad/s]');
bottomLegend(h, {'$c_D = 0.8$ m', '$c_D = 1.5$ m'});
saveFigure(f, figDir, 'fig08_yaw_rate_per_policy', saveFigures);

%% fig09 - angular rate RMSE per axis versus yaw-rate weight (c_M 0.25, c_D 0.8)
Y9 = M(M.thrust_to_weight == 1.9 & M.moment_scale == 0.25 & M.denominator == 0.8, :);
weights = unique(round(Y9.w_psi, 2))';
f = newFigure(S, 8, 6.5);
tl = tiledlayout(f, 1, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
ax = nexttile(tl); hold(ax, 'on');
rates = {'w_x_rms', 'w_y_rms', 'w_z_rms'};
offs = [-0.2 0 0.2];
h = gobjects(0);
for k = 1:3
    for i = 1:numel(weights)
        v = Y9.(rates{k})(abs(Y9.w_psi - weights(i)) < 1e-6);
        hh = seedStrip(ax, i + offs(k), v, S.axisColor(k, :), S.axisMarker{k});
        if i == 1, h(k) = hh; end
    end
end
formatCategoryAxis(ax, S, weights, 'Effective yaw-rate weight $w_\psi$');
ylabel(ax, 'Angular rate RMSE [rad/s]');
bottomLegend(h, {'$\omega_x$', '$\omega_y$', '$\omega_z$'});
saveFigure(f, figDir, 'fig09_angular_rate_vs_yaw_weight', saveFigures);

%% fig10 - yaw-penalty episode: same goal and seed for the three weights
exps = ["ms025_den08", "ms025_den08_yaw011", "ms025_den08_yaw101"];
rows = {'err', 'wx', 'wy', 'wz'};
ylab = {'$\|\mathbf{e}_p\|$ [m]', '$\omega_x$ [rad/s]', '$\omega_y$ [rad/s]', '$\omega_z$ [rad/s]'};
f = newFigure(S, 15, 13);
tl = tiledlayout(f, 4, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
for r = 1:4
    ax = nexttile(tl); hold(ax, 'on');
    h = gobjects(0);
    for i = 1:3
        Ei = EP(EP.thrust_to_weight == 1.9 & EP.experiment == exps(i) & EP.seed == yawSeed, :);
        h(i) = plot(ax, Ei.t, Ei.(rows{r}), S.weightLine{i}, 'Color', S.weightColor(i, :), 'LineWidth', 1.3);
    end
    if r > 1, yline(ax, 0, ':', 'Color', [0.4 0.4 0.4], 'HandleVisibility', 'off'); end
    xlim(ax, [0 10]); ylabel(ax, ylab{r}); formatAxis(ax, S);
    if r < 4, ax.XTickLabel = []; else, xlabel(ax, 'Time [s]'); end
    if r == 1
        legend(ax, h, compose('$w_\\psi = %.2f$', [0.01 0.11 1.01]), 'Location', 'northeast', 'Box', 'off', 'Interpreter', 'latex', 'FontSize', 9);
    end
end
saveFigure(f, figDir, sprintf('fig10_yaw_penalty_episode_seed%d', yawSeed), saveFigures);

%% fig11 - thrust action and thrust for k_T = 1.9, 3.9 and 3.9 with the hover at a0 = 0 (same configuration, goal, seed)
f = newFigure(S, 15, 8.5);
tl = tiledlayout(f, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
kts  = [1.9 3.9 3.9];
exps = ["ms025_den08_yaw101" "ms025_den08_yaw101" "ms025_den08_yaw101_hc"];
h = gobjects(0);
for r = 1:2
    ax = nexttile(tl); hold(ax, 'on');
    for i = 1:3
        Ei = EP(EP.thrust_to_weight == kts(i) & EP.experiment == exps(i) & EP.seed == ktSeed, :);
        if r == 1
            h(i) = plot(ax, Ei.t, Ei.a0, '-', 'Color', S.ktColor(i, :), 'LineWidth', 1.3);
            yline(ax, Ei.a0_hover(1), '--', 'Color', S.ktColor(i, :), 'LineWidth', 1, 'HandleVisibility', 'off');
        else
            plot(ax, Ei.t, Ei.thrust_N ./ Ei.weight_N, '-', 'Color', S.ktColor(i, :), 'LineWidth', 1.3);
        end
    end
    xlim(ax, [0 10]); formatAxis(ax, S);
    if r == 1
        ylabel(ax, 'Thrust action $a_0$'); ylim(ax, [-1 1]); ax.XTickLabel = [];
        h(4) = plot(ax, nan, nan, '--', 'Color', [0.3 0.3 0.3]);
    else
        ylabel(ax, '$T/(mg)$'); xlabel(ax, 'Time [s]');
        yline(ax, 1, ':', 'Color', [0.4 0.4 0.4], 'HandleVisibility', 'off');
    end
end
bottomLegend(h, {'$k_T = 1.9$', '$k_T = 3.9$', '$k_T = 3.9$, hover at $a_0 = 0$', 'hover action'}, 2);
saveFigure(f, figDir, 'fig11_thrust_action_kT', saveFigures);

%% console summary (median over seeds)
disp(groupsummary(M, {'thrust_to_weight', 'experiment'}, 'median', ...
    {'survival', 'final_err_m', 'rmse_m', 't20_s', 'w_z_rms'}));
end

%% =============================================================== helpers
function restoreDefaults(names, values)
for k = 1:numel(names), set(groot, names{k}, values{k}); end
end

function q = quantiles(Y, p)
% Quantiles of each row of Y (linear interpolation; no Statistics Toolbox needed).
Y = sort(Y, 2);
n = size(Y, 2);
pos = 1 + p * (n - 1);
lo = floor(pos); hi = ceil(pos); w = pos - lo;
q = Y(:, lo) .* (1 - w) + Y(:, hi) .* w;
end

function S = style()
S.font = 9;
S.scaleColor  = [0.80 0.47 0.13; 0.25 0.55 0.30; 0.10 0.30 0.65; ...  % c_M 0.01, 0.06, 0.12,
                 0.49 0.18 0.56; 0.75 0.15 0.20; 0.25 0.25 0.25];   %     0.25, 0.5, 1.0
S.denColor    = [0.10 0.30 0.65; 0.85 0.33 0.10];                   % c_D 0.8, 1.5
S.denMarker   = {'o', 's'};
S.denOffset   = [-0.17 0.17];
S.axisColor   = [0.00 0.45 0.74; 0.85 0.33 0.10; 0.47 0.67 0.19];   % x, y, z
S.axisMarker  = {'s', 'd', 'o'};
S.weightColor = [0.55 0.55 0.55; 0.85 0.55 0.10; 0.10 0.30 0.65];   % w_psi 0.01, 0.11, 1.01
S.weightLine  = {'-', '--', '-'};
S.ktColor     = [0.10 0.30 0.65; 0.85 0.33 0.10; 0.25 0.55 0.30];   % k_T 1.9, 3.9, 3.9 hover-centred
S.response    = [0.10 0.30 0.65];
end

function f = newFigure(S, widthCm, heightCm)
f = figure('Units', 'centimeters', 'Color', 'w', 'WindowStyle', 'normal', 'Resize', 'off');
f.Position = [2 2 widthCm heightCm];
f.UserData.sizeCm = [widthCm heightCm];               % restored before saving (the window manager may resize it)
set(f, 'DefaultAxesFontSize', S.font);
end

function [f, tl] = twoPanels(S)
f = newFigure(S, 15, 6.8);
tl = tiledlayout(f, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
end

function formatAxis(ax, S)
set(ax, 'FontName', 'Computer Modern', 'FontSize', S.font, 'TickLabelInterpreter', 'latex', 'Box', 'on', 'TickDir', 'in', 'LineWidth', 0.6);
grid(ax, 'on'); ax.GridAlpha = 0.15;
end

function formatCategoryAxis(ax, S, values, label)
formatAxis(ax, S);
set(ax, 'XTick', 1:numel(values), 'XTickLabel', compose('%.2f', values));
xlim(ax, [0.5 numel(values) + 0.5]); xlabel(ax, label); ax.XGrid = 'off';
end

function panelTitle(ax, d, den)
title(ax, sprintf('(%s) $c_D = %.1f$ m', char('a' + d - 1), den), 'FontWeight', 'normal');
end

function bottomLegend(h, names, nCols)
if nargin < 3, nCols = numel(names); end
lgd = legend(h, names, 'Orientation', 'horizontal', 'Box', 'off', 'Interpreter', 'latex', 'FontSize', 9, ...
    'NumColumns', nCols);
lgd.Layout.Tile = 'south';
end

function [x, Y] = bySeed(Tab, xField, yField, x, smoothWin)
% One column per seed, resampled on x (seeds may log different numbers of points).
seeds = unique(Tab.seed);
Y = nan(numel(x), numel(seeds));
for s = 1:numel(seeds)
    r = Tab(Tab.seed == seeds(s), :);
    [xs, first] = unique(r.(xField));
    ys = r.(yField)(first);
    ok = ~isnan(ys);
    if smoothWin > 1, ys(ok) = smoothdata(ys(ok), 'movmean', smoothWin); end
    Y(:, s) = interp1(xs(ok), ys(ok), x, 'linear', NaN);
end
end

function h = medianBand(ax, x, Y, color)
% Median over seeds (line) and interquartile range (shaded).
valid = all(~isnan(Y), 2);
x = x(valid); Y = Y(valid, :);
q = quantiles(Y, [0.25 0.5 0.75]);
fill(ax, [x; flipud(x)], [q(:, 1); flipud(q(:, 3))], color, 'FaceAlpha', 0.2, 'EdgeColor', 'none', ...
    'HandleVisibility', 'off');
h = plot(ax, x, q(:, 2), '-', 'Color', color, 'LineWidth', 1.4);
end

function h = seedStrip(ax, x0, v, color, marker)
% Individual seeds (open markers) with the median (filled) and interquartile range (bar).
x = x0 + linspace(-0.06, 0.06, numel(v))';
scatter(ax, x, v, 14, color, marker, 'HandleVisibility', 'off');
q = quantiles(v(:)', [0.25 0.5 0.75]);
plot(ax, [x0 x0], q([1 3]), '-', 'Color', color, 'LineWidth', 1.6, 'HandleVisibility', 'off');
h = plot(ax, x0, q(2), marker, 'Color', color, 'MarkerFaceColor', color, 'MarkerSize', 7);
end

function episodeFigure(S, E, fields, refFields, ylabels, names, limits, figDir, name, doSave)
% One subplot per axis: signal (solid), reference (dashed), limits (dotted).
n = numel(fields);
f = newFigure(S, 15, 10);
tl = tiledlayout(f, n, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
for k = 1:n
    ax = nexttile(tl); hold(ax, 'on');
    h = gobjects(0);
    y = E.(fields{k});
    h(1) = plot(ax, E.t, y, '-', 'Color', S.response, 'LineWidth', 1.4);
    yAll = y;
    if k <= numel(refFields) && ~isempty(refFields{k})
        r = E.(refFields{k});
        h(end + 1) = plot(ax, E.t, r, '--', 'Color', 'k', 'LineWidth', 1.1); %#ok<AGROW>
        yAll = [yAll; r]; %#ok<AGROW>
    end
    if ~isempty(limits) && ~isempty(limits{k})
        for v = limits{k}
            h(end + 1) = yline(ax, v, ':', 'Color', [0.45 0.45 0.45], 'LineWidth', 1.1); %#ok<AGROW>
        end
        yAll = [yAll; limits{k}(:)]; %#ok<AGROW>
    end
    lo = min(yAll); hi = max(yAll); span = max(hi - lo, 1e-3);
    top = 0.08;
    if k == 1, top = 0.80; end            % head-room: the legend sits above the data and the limit lines
    ylim(ax, [lo - 0.08 * span, hi + top * span]);
    xlim(ax, [0 10]); ylabel(ax, ylabels{k}); formatAxis(ax, S);
    if k < n, ax.XTickLabel = []; else, xlabel(ax, 'Time [s]'); end
    if k == 1
        keep = min(numel(h), numel(names));
        legend(ax, h(1:keep), names(1:keep), 'Location', 'northeast', 'Box', 'off', 'Interpreter', 'latex', 'FontSize', 9);
    end
end
saveFigure(f, figDir, name, doSave);
end

function saveFigure(f, figDir, name, doSave)
% PNG with exportgraphics; PDF with print -painters (same path as the manual export of the figure window):
% the LaTeX text becomes vector outlines instead of selectable text set in a substitute font.
if ~doSave, return; end
drawnow;
if ~isvalid(f), warning('Figure %s was closed before it could be saved.', name); return; end
sz = f.UserData.sizeCm;
set(f, 'Units', 'centimeters', 'Position', [2 2 sz]); drawnow;
exportgraphics(f, fullfile(figDir, [name '.png']), 'Resolution', 300);
set(f, 'PaperUnits', 'centimeters', 'PaperSize', sz, 'PaperPosition', [0 0 sz]);
print(f, fullfile(figDir, [name '.pdf']), '-dpdf', '-painters');
end

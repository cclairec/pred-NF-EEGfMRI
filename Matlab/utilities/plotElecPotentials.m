function [varFig,f1] = plotElecPotentials(EMap,potentials)
%
% <The RCSP (Regularized Common Spatial Pattern) Toolbox.>
%     Copyright (C) 2010  Fabien LOTTE
% 
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
% Visualization function for electrode potentials (using interpolation)
%   by Fabien Lotte (fprlotte@i2r.a-star.edu.sg)
%
%Input:
%EMAP: Map that specifies electrode location
%potentials: column vector containing the electrode potentials (instantaneous)

    X=cell2mat(EMap(2,:));
    Y=cell2mat(EMap(3,:));
    Z1=potentials';
    
    maxX = max(X);
    minX = min(X);
    maxY = max(Y);
    minY = min(Y);
    
    % Normalize to [-1 1] for plot (don't normalize)
    %maxZ=max(abs(Z1));
    %Z1=Z1./maxZ;

    % Head, nose and ear constants used in the plot
    rmax          = 1.0;                 % head radius
    CIRCGRID      = 201;                 % number of angles to use in drawing circles
    hwidth        = 0.007;               % width of head ring 
    hin           = rmax*(1- hwidth/2);  % inner head ring radius
    base          = rmax-.0046;
    basex         = 0.18*rmax;           % nose width
    tip           = 1.15*rmax; 
    tiphw         = .04*rmax;            % nose tip half width
    tipr          = .01*rmax;            % nose tip rounding
    q     = .04;                         % ear length
    EarX  = [.497-.005  .510  .518  .5299 .5419  .54    .547   .532   .510   .489-.005]*2; 
    EarY  = [q+.0555 q+.0775 q+.0783 q+.0746 q+.0555 -.0055 -.0932 -.1313 -.1384 -.1199]*2;

    function plotheadears()
        % plot head
        patch(headx,heady,ones(size(headx)),'k','edgecolor','k');
        % plot nose
        plot3([basex;tiphw;0;-tiphw;-basex],[base;tip-tipr;tip;tip-tipr;base],...
             2*ones(size([basex;tiphw;0;-tiphw;-basex])),...
             'Color','k','LineWidth',2);
        % plot left ear
        plot3(EarX,EarY,2*ones(size(EarX)),'color','k','LineWidth',2);    
        % plot right ear
        plot3(-EarX,EarY,2*ones(size(EarY)),'color','k','LineWidth',2);   
    end

    % Compute indices for interpolation
    GRID_SCALE = 200;        % plot map on a 67X67 grid
    iiY = linspace(minX,maxX,GRID_SCALE);   % x-axis description (row vector)
    iiX = linspace(minY,maxY,GRID_SCALE);   % y-axis description (row vector)

    % Compute coordinates for drawing the head
    cnv = convhull(X,Y);
    cnvfac = round(CIRCGRID/length(cnv)); % spline interpolate the convex hull
    if cnvfac < 1, cnvfac=1; end;
    CIRCGRID = cnvfac*length(cnv);
    startangle = atan2(X(cnv(1)),Y(cnv(1)));
    circ = linspace(0+startangle,2*pi+startangle,CIRCGRID);
    rx = sin(circ); 
    ry = cos(circ); 
    headx = [[rx(:)' rx(1) ]*(hin+hwidth)  [rx(:)' rx(1)]*hin];
    heady = [[ry(:)' ry(1) ]*(hin+hwidth)  [ry(:)' ry(1)]*hin];

    f1=figure;
    pos=get(f1,'Position');
    set(f1,'Position',pos);

    % Left Tap
    h1=subplot('Position',[0,0,0.925,0.925]);
    [iX,iY,iiZ1]=griddata(X,Y,Z1,iiY',iiX,'v4'); % interpolate data
       
    %mask points outside the convex-hull (as the interpolation would be
    %poor for these points)
    maskConvHull = inpolygon(iX,iY,X(cnv),Y(cnv));
    iiZ1(maskConvHull==0) = NaN; 
    
    [maxZCol maxZIndexCol] = max(iiZ1);
    [minZCol minZIndexCol] = min(iiZ1);
    [maxZRow maxZIndexRow] = max(maxZCol);
    [minZRow minZIndexRow] = min(minZCol);
            
    contourf(h1,iX,iY,iiZ1,20,'LineColor','None');       % plot the interpolated potentials
    
    title('Potentials');
    hold on;
    plotheadears();                                   % plot the head, nose, ears
    plot(X,Y,'o','MarkerEdgeColor','k',...
        'MarkerFaceColor','k','MarkerSize',2);        % plot the electrodes
    enames=cellstr(EMap(1,:));
    text(X,Y,enames);
    axis square;
    xlim([-1.2 1.2]);
    ylim([-1 1.2]);
    hold off;
    axis off;
    
    % Plot the colorbar and move it right
%     h3=colorbar('East');
%     pos3=get(h3,'Position');
%     pos3(1)=pos3(1)+0.075;
%     set(h3,'Position',pos3);
    varFig = get(gca,'children');
    %close(f1)
end
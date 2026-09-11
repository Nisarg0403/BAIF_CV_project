import React, { useEffect, useRef } from 'react';

export default function CattleCanvas({ 
  imageUrl, 
  landmarks, 
  measurements, 
  contourPoints,
  viewType = 'left',
  showContour = true,
  showLandmarks = true,
  showVectors = true 
}) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!imageUrl || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const container = canvas.parentElement;

    // Use fixed container height 480px and measured width
    const containerW = container ? container.clientWidth || 700 : 700;
    const containerH = 480;

    canvas.width = containerW;
    canvas.height = containerH;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = imageUrl;

    img.onload = () => {
      // Clear background with dark slate canvas color
      ctx.fillStyle = '#0F172A';
      ctx.fillRect(0, 0, containerW, containerH);

      const imgW = img.width;
      const imgH = img.height;

      // Calculate aspect-ratio preserving fit (Letterbox / Contain)
      const imgAspect = imgW / imgH;
      const containerAspect = containerW / containerH;

      let drawW, drawH, drawX, drawY;
      if (imgAspect > containerAspect) {
        drawW = containerW;
        drawH = containerW / imgAspect;
        drawX = 0;
        drawY = (containerH - drawH) / 2;
      } else {
        drawH = containerH;
        drawW = containerH * imgAspect;
        drawX = (containerW - drawW) / 2;
        drawY = 0;
      }

      // 1. Draw original image scaled inside letterbox
      ctx.drawImage(img, drawX, drawY, drawW, drawH);

      const scaleX = drawW / imgW;
      const scaleY = drawH / imgH;

      // Coordinate transformation: Image space (0..imgW, 0..imgH) -> Canvas space (drawX..drawX+drawW, drawY..drawY+drawH)
      const toX = (px) => drawX + px * scaleX;
      const toY = (py) => drawY + py * scaleY;

      // 2. Draw Translucent Silhouette Contour
      if (showContour && contourPoints && contourPoints.length > 0) {
        ctx.beginPath();
        ctx.moveTo(toX(contourPoints[0][0]), toY(contourPoints[0][1]));
        for (let i = 1; i < contourPoints.length; i++) {
          ctx.lineTo(toX(contourPoints[i][0]), toY(contourPoints[i][1]));
        }
        ctx.closePath();

        // Translucent emerald green fill over cow silhouette
        ctx.fillStyle = 'rgba(16, 185, 129, 0.22)';
        ctx.fill();

        // Dashed white outline tracing exact cow edge
        ctx.setLineDash([6, 5]);
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Helper function to render metric pill badges
      const drawBadge = (text, label, cx, cy) => {
        ctx.font = '600 12px "Inter", sans-serif';
        const textWidth = ctx.measureText(`${label}: ${text}`).width;
        const padX = 10;
        const padY = 5;
        const badgeW = textWidth + padX * 2;
        const badgeH = 26;

        const bx = Math.max(10, Math.min(containerW - badgeW - 10, cx - badgeW / 2));
        const by = Math.max(10, Math.min(containerH - badgeH - 10, cy - badgeH / 2));

        // Badge Background
        ctx.fillStyle = 'rgba(15, 23, 42, 0.90)';
        ctx.beginPath();
        ctx.roundRect(bx, by, badgeW, badgeH, 6);
        ctx.fill();

        // Badge Border
        ctx.strokeStyle = 'rgba(16, 185, 129, 0.6)';
        ctx.lineWidth = 1.2;
        ctx.stroke();

        // Label Prefix
        ctx.fillStyle = '#94A3B8';
        ctx.font = '500 11px "Inter", sans-serif';
        ctx.fillText(label, bx + padX, by + padY + 11);

        // Value String
        ctx.fillStyle = '#34D399';
        ctx.font = '700 12px "Inter", sans-serif';
        const labelW = ctx.measureText(`${label} `).width;
        ctx.fillText(text, bx + padX + labelW, by + padY + 11);
      };

      // 3. Draw Measurement Vectors & Caliper Lines
      if (showVectors && measurements) {
        const isRear = viewType === 'rear' || (landmarks && landmarks.B_left && landmarks.B_right);

        if (isRear && landmarks && landmarks.B_left && landmarks.B_right) {
          // Rear View Caliper (Rump Width)
          const pL = { x: toX(landmarks.B_left.x), y: toY(landmarks.B_left.y) };
          const pR = { x: toX(landmarks.B_right.x), y: toY(landmarks.B_right.y) };

          ctx.setLineDash([6, 4]);
          ctx.strokeStyle = '#34D399';
          ctx.lineWidth = 2.5;
          ctx.beginPath();
          ctx.moveTo(pL.x, pL.y);
          ctx.lineTo(pR.x, pR.y);
          ctx.stroke();
          ctx.setLineDash([]);

          if (measurements.chest_girth_cm) {
            const rumpW = (measurements.chest_girth_cm * 0.26).toFixed(1);
            drawBadge(`${rumpW} cm`, 'Rump Width', (pL.x + pR.x) / 2, pL.y - 20);
          }
        } else if (landmarks && landmarks.D && landmarks.C && landmarks.A && landmarks.E1) {
          // Side Profile Caliper Lines settled directly on landmark points
          const pD = { x: toX(landmarks.D.x), y: toY(landmarks.D.y) };   // Shoulder Joint
          const pC = { x: toX(landmarks.C.x), y: toY(landmarks.C.y) };   // Pin Bone
          const pA = { x: toX(landmarks.A.x), y: toY(landmarks.A.y) };   // Withers (Shoulder Top)
          const pE1 = { x: toX(landmarks.E1.x), y: toY(landmarks.E1.y) }; // Front Hoof Base

          // Body Length Caliper Vector (Point D to Point C)
          ctx.setLineDash([6, 4]);
          ctx.strokeStyle = '#38BDF8'; // Sky blue for Body Length
          ctx.lineWidth = 2.5;
          ctx.beginPath();
          ctx.moveTo(pD.x, pD.y);
          ctx.lineTo(pC.x, pC.y);
          ctx.stroke();

          // Withers Height / Chest Girth Vector (Point A to Point E1)
          ctx.strokeStyle = '#34D399'; // Emerald green for Height/Girth
          ctx.beginPath();
          ctx.moveTo(pA.x, pA.y);
          ctx.lineTo(pE1.x, pE1.y);
          ctx.stroke();
          ctx.setLineDash([]);

          // Draw Badges at Midpoints
          if (measurements.body_length_cm) {
            drawBadge(`${measurements.body_length_cm} cm`, 'Body Length', (pD.x + pC.x) / 2, (pD.y + pC.y) / 2 - 16);
          }
          if (measurements.chest_girth_cm) {
            drawBadge(`${measurements.chest_girth_cm} cm`, 'Chest Girth', pA.x + 15, (pA.y + pE1.y) / 2);
          }
        }
      }

      // 4. Draw Glowing Landmark Target Dots (A, B, C, D, E1, E2, F, G)
      if (showLandmarks && landmarks) {
        Object.entries(landmarks).forEach(([name, pt]) => {
          if (!pt || typeof pt.x !== 'number' || typeof pt.y !== 'number') return;

          const cx = toX(pt.x);
          const cy = toY(pt.y);

          // Outer glowing green ring
          ctx.beginPath();
          ctx.arc(cx, cy, 7, 0, 2 * Math.PI);
          ctx.fillStyle = '#10B981';
          ctx.fill();

          // Inner solid white dot
          ctx.beginPath();
          ctx.arc(cx, cy, 3.5, 0, 2 * Math.PI);
          ctx.fillStyle = '#FFFFFF';
          ctx.fill();

          // Point Label Tag
          ctx.font = '700 11px "Inter", sans-serif';
          ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
          const textW = ctx.measureText(name).width;
          ctx.fillRect(cx + 8, cy - 11, textW + 6, 15);

          ctx.fillStyle = '#FFFFFF';
          ctx.fillText(name, cx + 11, cy);
        });
      }
    };
  }, [imageUrl, landmarks, measurements, contourPoints, viewType, showContour, showLandmarks, showVectors]);

  return (
    <div className="visualizer-container">
      <canvas ref={canvasRef} className="visualizer-canvas" />
    </div>
  );
}


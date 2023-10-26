#!/bin/bash
mkdir checkpoints

echo "downloading from S3 ..."

[[ ! -f "BoostingMonocularDepth/pix2pix/checkpoints/mergemodel/latest_net_G.pth" ]] && wget -P BoostingMonocularDepth/pix2pix/checkpoints/mergemodel/ https://teads-ai-creative-lab-weights.s3.eu-west-1.amazonaws.com/checkpoints/mergemodel/latest_net_G.pth
[[ ! -f "checkpoints/color-model.pth" ]] && wget -P checkpoints/ https://teads-ai-creative-lab-weights.s3.eu-west-1.amazonaws.com/checkpoints/color-model.pth
[[ ! -f "checkpoints/depth-model.pth" ]] && wget -P checkpoints/ https://teads-ai-creative-lab-weights.s3.eu-west-1.amazonaws.com/checkpoints/depth-model.pth
[[ ! -f "checkpoints/edge-model.pth" ]] && wget -P checkpoints/ https://teads-ai-creative-lab-weights.s3.eu-west-1.amazonaws.com/checkpoints/edge-model.pth
[[ ! -f "MiDaS/model.pt" ]] && get -P MiDaS/ https://teads-ai-creative-lab-weights.s3.eu-west-1.amazonaws.com/MiDaS/model.pt
[[ ! -f "weights/ig_resnext101_32x8-c38310e5.pth" ]] && wget -P weights/ https://teads-ai-creative-lab-weights.s3.eu-west-1.amazonaws.com/weights/ig_resnext101_32x8-c38310e5.pth
[[ ! -f "BoostingMonocularDepth/midas/model.pt" ]] && wget -P BoostingMonocularDepth/midas/ https://teads-ai-creative-lab-weights.s3.eu-west-1.amazonaws.com/midas/model.pt

echo "Done`"

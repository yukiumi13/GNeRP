PREFIX=$1
CAMERA=$2
cd ~/neurecon
python -m tools.render_view --downscale 4 --config ${PREFIX}/config.yaml \
--load_pt ${PREFIX}/ckpts/00100000.pt \
--camera_path ${CAMERA} \
--num_views 360 \
--device_ids 0,1,2,3
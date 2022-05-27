# GIMP helper script (Python 2)
## This script saves each channel as a separate RGBA file.
## Each channel should provide the selection mask for a papyrus fragment.

original_image = gimp.image_list()[0]
interlace, compression, bkgd, gama, offs, phys, time, comment, svtrans = pdb.file_png_get_defaults()
n_fragments = len(original_image.channels)
for i in range(n_fragments):
    image = pdb.gimp_image_duplicate(original_image)
    display = pdb.gimp_display_new(image)
    channel = image.channels[i]
    pdb.gimp_image_select_item(image, 2, channel)
    non_empty, x1, y1, x2, y2 = pdb.gimp_selection_bounds(image)
    pdb.gimp_image_crop(image, x2 - x1, y2 - y1, x1, y1)
    layer = image.layers[0]
    mask = layer.create_mask(ADD_SELECTION_MASK)
    layer.add_mask(mask)
    pdb.gimp_image_remove_layer_mask(image, layer, MASK_APPLY)
    out = '{}.png'.format(channel.name)
    pdb.file_png_save2(image, layer, out, out, interlace, compression, bkgd, gama, offs, phys, time, comment, svtrans)
    pdb.gimp_display_delete(display)
import { Niivue, DRAG_MODE, SLICE_TYPE, MULTIPLANAR_TYPE, SHOW_RENDER } from '@niivue/niivue'
import * as ort from "./node_modules/onnxruntime-web/dist/ort.all.mjs"

const nv = new Niivue()
const isiPad = navigator.userAgent.match(/iPad/i) != null

const onnxOptions = {
  executionProviders: [
    {
      name: 'webgpu',
    },
    {
      name: 'wasm',
    }
  ],
  graphOptimizationLevel: 'disabled'
} // n.b. in future graphOptimizationLevel extended

const onClipChange = (e) => {
  if (e.target.checked) {
    nv.setClipPlane([0, 0, 90])
  } else {
    nv.setClipPlane([2, 0, 90])
  }
}

const onBackOpacityChange = (e) => {
  nv.setOpacity(0, e.target.value / 255)
  nv.updateGLVolume()
}

const onOverlayOpacityChange = (e) => {
  nv.setOpacity(1, e.target.value / 255)
  nv.updateGLVolume()
}

const onDrawingOpacityChange = (e) => {
  nv.drawOpacity = e.target.value / 255
  nv.drawScene()
}

const onImageLoaded = async () => {
  // if more than one image aleady, then just return. We only run
  // this if the user drags and drops a new image.
  if (nv.volumes.length > 1) {
    return
  }
  // check input image dimensions. If they are not equal then 
  // give the user a diaglog to conform the image.
  const inDims = nv.volumes[0].dims
  const isEqualDims = inDims[1] === inDims[2] && inDims[2] === inDims[3]
  // if not equal then give the user a dialog to conform the image
  if (!isEqualDims) {
    const userConfirmed = window.confirm('The input image dimensions are not equal. Would you like to conform the image?')
    if (userConfirmed) {
      // fire off the conform check event 
      conform.checked = true
      await onConformChange({ target: { checked: true } })
    } else {
      window.alert('Please conform the image before proceeding. The segmentation will not work correctly.')
    }
  } else {
    // clone the loaded image to create an overlay that stores the segmentation
    const overlay = nv.cloneVolume(0)
    overlay.img.fill(0) // fill with zeros since it will hold binary segmentation later
    overlay.opacity = 0.8
    // add the overlay to niivue
    nv.addVolume(overlay)
  }
  // set drawing enabled to true
  // if ipad, set to false by default
  // if (isiPad) {
  //   nv.setDrawingEnabled(false)
  // } else {
  //   nv.setDrawingEnabled(true)
  // }
  nv.setDrawingEnabled(true)
}

const onConformChange = async (e) => {
  if (nv.volumes.length < 1) {
    return
  }
  if (e.target.checked) {
    closeAllOverlays()
    await ensureConformed()
    nv.closeDrawing()
    nv.createEmptyDrawing()
    // clone the loaded image to create an overlay that stores the segmentation
    const overlay = nv.cloneVolume(0)
    overlay.img.fill(0) // fill with zeros since it will hold binary segmentation later
    overlay.opacity = 0.5
    // add the overlay to niivue
    nv.addVolume(overlay)
  }
}

const onSaveImageClick = () => {
  nv.volumes[1].saveToDisk('segmentation.nii')
}

const ensureConformed = async () => {
  const nii = nv.volumes[0]
  let isConformed = nii.dims[1] === 256 && nii.dims[2] === 256 && nii.dims[3] === 256
  if (nii.permRAS[0] !== -1 || nii.permRAS[1] !== 3 || nii.permRAS[2] !== -2) {
    isConformed = false
  }
  if (isConformed) {
    return
  }
  const nii2 = await nv.conform(nii, true, true, false)
  nv.removeVolume(nv.volumes[0])
  nv.addVolume(nii2)
}

const closeAllOverlays = () => {
  while (nv.volumes.length > 1) {
    nv.removeVolume(nv.volumes[1])
  }
}

const showLoadingCircle = () => {
  loadingCircle.classList.remove('hidden')
}

const hideLoadingCircle = () => {
  loadingCircle.classList.add('hidden')
}

const getSlice = (volIdx, start, end) => {
  const slice = nv.volumes[volIdx].getVolumeData(start, end)
  const data = slice[0]
  const dims = slice[1]
  const axcorsag = nv.tileIndex(...nv.mousePos)
  const ax = 0
  const cor = 1
  const sag = 2 
  // dims will hold H and W, but we need to determine which is which based
  // on the plane the user is drawing on.
  // if the user is drawing on the axial plane, then the first dimension is W
  // and the second dimension is H.
  // if the user is drawing on the coronal plane, then the first dimension is W
  // and the third dimension is H
  // if the user is drawing on the sagittal plane, then the second dimension is W
  // and the third dimension is H.
  // we need to determine which plane the user is drawing on, and then set the
  // dimensions accordingly.
  if (axcorsag === ax) {
    return { data, dims: [dims[0], dims[1]] }
  } else if (axcorsag === cor) {
    return { data, dims: [dims[0], dims[2]] }
  } else if (axcorsag === sag) {
    return { data, dims: [dims[2], dims[1]] }
  }
}

const copyDrawingToOverlay = (drawing, overlay) => {
  for (let i = 0; i < drawing.length; i++) {
    if (drawing[i] > 0) {
      overlay.img[i] = 1
    } else {
      overlay.img[i] = overlay.img[i] === 1 ? 1 : 0
    }
  }
}

const normalize = (img) => {
  //  TODO: ONNX not JavaScript https://onnx.ai/onnx/operators/onnx_aionnxml_Normalizer.html
  let mx = img[0]
  let mn = mx
  for (let i = 0; i < img.length; i++) {
    mx = Math.max(mx, img[i])
    mn = Math.min(mn, img[i])
  }
  let scale = 1 / (mx - mn)
  for (let i = 0; i < img.length; i++) {
    img[i] = (img[i] - mn) * scale
  }
}

// drawing pen can be 1, 2, 3, (e.g. red, green, blue)
const binarizeDrawing = (drawing) => {
  for (let i = 0; i < drawing.length; i++) {
    drawing[i] = drawing[i] > 0 ? 1 : 0
  }
}

const sigmoid = (x) => {
  return 1 / (1 + Math.exp(-x))
}

// when user is done making scribbles, run the segmentation
const doSegment = async (start, end) => {
  // show loading circle
  showLoadingCircle()
  // background image index
  const backIdx = 0
  // get the slice data from the active slice the user was drawing on
  const inputSlice = getSlice(backIdx, start, end)
  // get W, H, D of the slice
  const [W, H] = inputSlice.dims
  // get reference to the overlay volume for later
  const overlay = nv.volumes[1]
  // make sure slice data is a Float32Array
  const inputSlice32 = new Float32Array(inputSlice.data)
  // get reference to drawing from user clicks
  const drawing = nv.drawBitmap
  // is there a drawing? detect if any elements are greater than 0
  const hasDrawing = drawing.some(v => v > 0)
  // if no drawing, then return
  if (!hasDrawing) {
    hideLoadingCircle()
    return
  }
  copyDrawingToOverlay(drawing, overlay)
  // get the the slice data from the drawing overlay that 
  // has now been populated from the drawBitmap (which came from user clicks)
  const drawingSlice = getSlice(1, start, end)
  const drawingSlice32 = new Float32Array(drawingSlice.data)
  // binarize the drawing
  binarizeDrawing(drawingSlice32)
  // normalize input data to range 0..1
  normalize(inputSlice32) // no return value since it does this in place

  // create onnx runtime session for running inference
  const session = await ort.InferenceSession.create(
    './scribbleprompt_unet.onnx',
    onnxOptions
  )
  // expected scribble shape =  [B, 5, H, W]
  const HW = H * W
  const shape = [1, 5, H, W]
  const componentShape = [1, 1, H, W]
  // create tensor with correct shape filled with zeros
  const nvox = shape.reduce((a, b) => a * b)
  const inputTensor = new ort.Tensor('float32', new Float32Array(nvox), shape)
  // make 5 tensors: img, box, click, scribble, mask
  // img is the input slice data from the background image
  const imgTensor = new ort.Tensor('float32', inputSlice32, componentShape)
  // box not used
  const boxTensor = new ort.Tensor('float32', new Float32Array(HW).fill(0), componentShape)
  // clicks are the user clicks (the drawing from niivue)
  const clickTensor = new ort.Tensor('float32', drawingSlice32, componentShape)
  // scribble is not used. Not sure what the difference between clicks and scribble is,
  // but using drawingSlice32 for scribbles does not work as expected.
  const scribbleTensor = new ort.Tensor('float32', new Float32Array(HW).fill(0), componentShape)
  // mask is the previous mask from scribble inference. Not used here, but perhaps could be
  const maskTensor = new ort.Tensor('float32', new Float32Array(HW).fill(0), componentShape)

  // now concatenate the tensors for input to the model
  inputTensor.data.set(imgTensor.data, 0)
  inputTensor.data.set(boxTensor.data, HW)
  inputTensor.data.set(clickTensor.data, 2 * HW)
  inputTensor.data.set(scribbleTensor.data, 3 * HW)
  inputTensor.data.set(maskTensor.data, 4 * HW)
  const modelInputs = { "input": inputTensor }

  // run onnx inference
  // get time stamp for timing
  const t0 = performance.now()
  const results = await session.run(modelInputs)
  const t1 = performance.now()
  const inferenceTime = t1 - t0
  console.log(`inference time: ${inferenceTime} ms`)

  // output has shape [1, 1, H, W]
  const outSliceData = results.output.cpuData
  const threshold = 0.5
  // model returns logits, so we need to apply sigmoid.
  for (let i = 0; i < outSliceData.length; i++) {
    outSliceData[i] = sigmoid(outSliceData[i])
    outSliceData[i] = outSliceData[i] < threshold ? 0 : 1
    // combine the segmentation with the the original drawing
    if (drawingSlice32[i] === 1) {
      outSliceData[i] = 1
    }
  }

  // update the overlay slice with the segmentation
  overlay.setVolumeData(start, end, outSliceData)
  // make sure the overlay colormap is red
  overlay.setColormap('red')
  nv.updateGLVolume()
  // reset the drawing for the next call
  nv.closeDrawing()
  nv.createEmptyDrawing()
  // hide the loading circle
  hideLoadingCircle()

}

const onSegmentClick = async () => {
  if (nv.volumes.length < 1) {
    window.alert('Please open a voxel-based image')
    return
  }
  const vox = nv.frac2vox(nv.scene.crosshairPos, 0)
  const dims = nv.volumes[0].dims
  // const start = [0, 0, vox[2]]
  // const end = [dims[1] - 1, dims[2] - 1, vox[2]]
  const axcorsag = nv.tileIndex(...nv.mousePos)
  const ax = 0
  const cor = 1
  const sag = 2
  // determine which plane the user is drawing on, and set the start and end
  // to the full slice of that plane
  let start, end = [0, 0, 0]
  if (axcorsag === ax) {
    start = [0, 0, vox[2]]
    end = [dims[1] - 1, dims[2] - 1, vox[2]]
  } else if (axcorsag === cor) {
    start = [0, vox[1], 0]
    end = [dims[1] - 1, vox[1], dims[3] - 1]
  } else if (axcorsag === sag) {
    start = [vox[0], 0, 0]
    end = [vox[0], dims[2] - 1, dims[3] - 1]
  }
  // run the segmentation and send the slice info
  await doSegment(start, end)
}

const handleLocationChange = (data) => {
  document.getElementById("intensity").innerHTML = data.string
}

async function main() {
  // set callback for clip plane checkbox
  clipCheck.onchange = onClipChange
  // set callback for back opacity slider
  opacitySlider0.onchange = onBackOpacityChange
  // set callback for overlay opacity slider
  opacitySlider1.onchange = onOverlayOpacityChange
  // set callback for drawing opacity slider
  opacitySlider2.onchange = onDrawingOpacityChange
  // set callback for save image button (saves the segmentation)
  saveImgBtn.onclick = onSaveImageClick
  // set callback for segment button
  segmentBtn.onclick = onSegmentClick
  // set callback for conform checkbox
  conform.onchange = onConformChange
  // bind "s" key up to onSegmentClick also
  document.addEventListener('keyup', (e) => {
    if (e.key === 's') {
      onSegmentClick()
    }
  })
  // bind shift + z to undo
  document.addEventListener('keyup', (e) => {
    if (e.key === 'z') {
      nv.drawUndo()
    }
  })
  // set callback for when the crosshair moves
  // to report the intensity of the voxel under the crosshair
  // and the location. 
  nv.onLocationChange = handleLocationChange
  // get canvas element
  const gl1 = document.getElementById('gl1')
  nv.attachToCanvas(gl1)
  nv.setDrawingEnabled(false)
  nv.opts.maxDrawUndoBitmaps = 30
  nv.opts.multiplanarForceRender = true
  nv.opts.yoke3Dto2DZoom = true
  nv.opts.crosshairGap = 5
  nv.setInterpolation(false) // use nearest neighbor
  nv.onImageLoaded = onImageLoaded
  await nv.loadVolumes([{ url: './stroke.nii.gz' }])
  nv.setPenValue(2, false)
  nv.drawOpacity = 1
  nv.setMultiplanarLayout(MULTIPLANAR_TYPE.GRID)
  nv.setSliceType(SLICE_TYPE.MULTIPLANAR)
  nv.opts.multiplanarShowRender = SHOW_RENDER.ALWAYS
  nv.opts.dragMode = DRAG_MODE.slicer3D

  // on mouse up dosegment
  window.addEventListener('pointerup', async function(event) {
    if (event.pointerType === 'pen') {
      await onSegmentClick()
    }
  })
  
}

main()

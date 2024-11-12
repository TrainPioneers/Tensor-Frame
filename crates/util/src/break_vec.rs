pub async fn break_vec(vec: Vec<f32>) -> Vec<Vec<f32>> {
    let chunk_size: usize = 262144;
    vec.chunks(chunk_size).map(|chunk| chunk.to_vec()).collect()
}